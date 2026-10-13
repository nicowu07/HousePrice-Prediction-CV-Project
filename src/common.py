"""Shared plumbing: data loading, metrics, target decoding, submission writing.

Everything that more than one experiment needs lives here, so that every
reported number is produced by the same code path and runs stay comparable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "house_dataset"
FEATURE_DIR = PROJECT_ROOT / "features"
SPLIT_PATH = PROJECT_ROOT / "data" / "splits" / "folds.csv"
SUBMISSION_DIR = PROJECT_ROOT / "outputs" / "submissions"
EXPERIMENT_DIR = PROJECT_ROOT / "outputs" / "experiments"

# The provided prices are filtered to this closed interval (p0=195, p100=2000,
# with only 10 and 3 rows respectively on the bounds, so this is a filter and
# not censoring). The test split is an arbitrary split of the same filtered
# pool, so a prediction outside the interval can only ever increase squared
# error. Clipping is therefore free and we apply it to every prediction.
PRICE_MIN = 195.0
PRICE_MAX = 2000.0

# Bands used for error analysis throughout. Chosen so the top two bands isolate
# the expensive tail that carries most of the squared error.
PRICE_BANDS = [(195, 400), (400, 700), (700, 1000), (1000, 1500), (1500, 2001)]


def load_folds() -> pd.DataFrame:
    """imageid, price, group, fold -- the frozen split."""
    if not SPLIT_PATH.exists():
        raise SystemExit(f"{SPLIT_PATH} missing - run src/make_splits.py first")
    return pd.read_csv(SPLIT_PATH)


def load_features(tag: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (features, imageid) for a cached extraction."""
    path = FEATURE_DIR / f"{tag}__{split}.npz"
    if not path.exists():
        available = sorted(p.name for p in FEATURE_DIR.glob("*.npz"))
        raise SystemExit(f"{path} missing. Available: {available}")
    data = np.load(path, allow_pickle=True)
    return data["features"], data["imageid"]


def align_features(tag: str, folds: pd.DataFrame) -> np.ndarray:
    """Load training features and reorder them to match the fold table."""
    features, imageid = load_features(tag, "train")
    order = {img: i for i, img in enumerate(imageid)}
    idx = np.array([order[img] for img in folds["imageid"]])
    return features[idx]


def clip_prices(pred: np.ndarray) -> np.ndarray:
    return np.clip(pred, PRICE_MIN, PRICE_MAX)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def calibration_slope(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Slope b of the OLS fit true = a + b * pred.

    This is the diagnostic that separates two very different situations that
    both look like "the model under-predicts expensive houses":

      b ~= 1  the predictor is calibrated. Regression toward the mean is the
              MSE-optimal response to limited signal, and inflating the high
              end would *increase* squared error.
      b > 1   the predictor is over-shrunk -- it is underfitting, and a
              monotone recalibration recovers error for free.

    Without this number, error analysis on the tail is guesswork.
    """
    return float(np.polyfit(y_pred, y_true, 1)[0])


def band_errors(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Per-price-band squared error, and each band's share of the total."""
    sq = (y_true - y_pred) ** 2
    rows = []
    for low, high in PRICE_BANDS:
        mask = (y_true >= low) & (y_true < high)
        if not mask.any():
            continue
        rows.append({
            "band": f"{low}-{high if high <= 2000 else 2000}",
            "n": int(mask.sum()),
            "pct_rows": 100 * mask.mean(),
            "mse": sq[mask].mean(),
            "bias": float((y_pred[mask] - y_true[mask]).mean()),
            "pct_total_se": 100 * sq[mask].sum() / sq.sum(),
        })
    return pd.DataFrame(rows)


def reliability(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Mean outcome against mean prediction, binned by *prediction*.

    This is the honest calibration test, and it is not the same as the
    per-true-price-band table. Conditioning on the true price makes even a
    Bayes-optimal predictor look biased -- expensive houses will always appear
    under-predicted, because among houses that happen to be expensive, the
    ones the model could not identify drag the average prediction down. That
    is regression toward the mean, not a defect.

    Conditioning on the *prediction* asks the question that actually matters:
    of the houses this model values at 1400, is the average sale price 1400?
    Points on the diagonal mean the predictor is calibrated and its shrinkage
    is correct; a systematic tilt means it is over- or under-confident and a
    monotone recalibration will reduce squared error.
    """
    order = np.argsort(y_pred)
    bins = np.array_split(order, n_bins)
    rows = []
    for i, idx in enumerate(bins):
        rows.append({
            "decile": i + 1,
            "n": len(idx),
            "mean_pred": float(y_pred[idx].mean()),
            "mean_true": float(y_true[idx].mean()),
            "gap": float(y_true[idx].mean() - y_pred[idx].mean()),
            "rmse": float(np.sqrt(((y_true[idx] - y_pred[idx]) ** 2).mean())),
        })
    return pd.DataFrame(rows)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    mse = float((err ** 2).mean())
    var = float(y_true.var())
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.abs(err).mean()),
        "r2": float(1 - mse / var),
        "bias": float(err.mean()),
        "slope": calibration_slope(y_true, y_pred),
        # Headline comparison: how much of the constant-prediction baseline's
        # error does this model remove?
        "pct_vs_baseline": float(100 * (1 - mse / var)),
    }


def format_metrics(name: str, m: dict[str, float]) -> str:
    return (f"{name:<28} MSE {m['mse']:>9,.0f}  RMSE {m['rmse']:>6.1f}  "
            f"MAE {m['mae']:>6.1f}  R2 {m['r2']:>6.3f}  "
            f"bias {m['bias']:>+7.1f}  slope {m['slope']:.3f}")


# --------------------------------------------------------------------------
# target transforms
# --------------------------------------------------------------------------

class TargetDecoder:
    """Maps a model's output back to an estimate of E[price | x].

    Fitted on out-of-fold predictions only. In-sample residuals are
    systematically too small, so a smearing factor or residual variance
    estimated from them would under-correct.
    """

    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.factor = 1.0
        self.sigma2 = 0.0
        self.iso = None

    @property
    def log_space(self) -> bool:
        return self.mode != "raw"

    def encode(self, price: np.ndarray) -> np.ndarray:
        return price if self.mode == "raw" else np.log(price)

    def fit(self, oof_pred: np.ndarray, y_true: np.ndarray) -> "TargetDecoder":
        if self.mode in ("raw", "log_naive"):
            return self
        if self.mode == "log_smearing":
            # Duan (1983): E[p|x] = exp(mu) * E[exp(residual)], estimated
            # non-parametrically, so it does not assume log-normal residuals.
            self.factor = float(np.exp(np.log(y_true) - oof_pred).mean())
        elif self.mode == "log_lognormal":
            # Analytic correction under log-normal residuals. The data support
            # it here: log price has skew 0.03.
            self.sigma2 = float((np.log(y_true) - oof_pred).var())
        elif self.mode == "log_isotonic":
            from sklearn.isotonic import IsotonicRegression

            self.iso = IsotonicRegression(out_of_bounds="clip").fit(np.exp(oof_pred), y_true)
        else:
            raise ValueError(f"unknown decode mode {self.mode!r}")
        return self

    def decode(self, pred: np.ndarray) -> np.ndarray:
        if self.mode == "raw":
            out = pred
        elif self.mode == "log_naive":
            out = np.exp(pred)
        elif self.mode == "log_smearing":
            out = np.exp(pred) * self.factor
        elif self.mode == "log_lognormal":
            out = np.exp(pred + self.sigma2 / 2)
        elif self.mode == "log_isotonic":
            out = self.iso.predict(np.exp(pred))
        else:
            raise ValueError(f"unknown decode mode {self.mode!r}")
        return clip_prices(out)


# --------------------------------------------------------------------------
# submission
# --------------------------------------------------------------------------

def write_submission(imageid: np.ndarray, price: np.ndarray, name: str) -> Path:
    """Write a Kaggle-format CSV, in the order sample_solution.csv uses."""
    sample = pd.read_csv(DATA_DIR / "sample_solution.csv")
    pred = pd.DataFrame({"imageid": imageid, "price": clip_prices(price)})
    out = sample[["imageid"]].merge(pred, on="imageid", how="left")
    if out["price"].isna().any():
        raise SystemExit(f"{int(out['price'].isna().sum())} test images have no prediction")
    assert out["imageid"].tolist() == sample["imageid"].tolist()

    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    path = SUBMISSION_DIR / f"{name}.csv"
    out.to_csv(path, index=False)
    print(f"wrote {path}  n={len(out)}  "
          f"mean {out['price'].mean():.1f}  min {out['price'].min():.1f}  max {out['price'].max():.1f}")
    return path
