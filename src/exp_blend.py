"""Prediction-level fusion over every out-of-fold prediction we have stored.

The project plan listed prediction-level fusion as a candidate alongside
feature concatenation; this is that experiment. Concatenating features lets
one head learn interactions between backbones, but it also forces a single
model to fit 11,000 dimensions from 8,000 rows. Blending instead keeps each
model's own fit and only learns a handful of mixing weights, which is a much
smaller estimation problem. Which wins is empirical.

Two correctness points, both of which would otherwise inflate the result:

*   **The oracle-routed predictions are excluded.** They were produced by
    routing on the *true* price tier, so they encode the answer. Including
    them would be leakage, not ensembling.
*   **Blend weights are fitted out-of-fold.** Every stored vector is already
    an out-of-fold prediction, but the weights themselves are a second fit on
    the same rows. For fold k the weights are solved on the other four folds
    and applied to fold k, so the reported MSE never sees weights chosen with
    its own rows. The difference is not cosmetic: with ~50 candidates, in-fold
    weight fitting is heavily optimistic.

Usage
-----
    python src/exp_blend.py --top 12
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from common import (
    EXPERIMENT_DIR,
    band_errors,
    clip_prices,
    load_folds,
    metrics,
    reliability,
)

# Produced by routing on the true tier; keeping them would be leakage.
EXCLUDE_KEYS = {"oracle"}


def collect(y: np.ndarray) -> tuple[np.ndarray, list[str]]:
    preds, names = [], []
    for path in sorted(glob.glob(str(EXPERIMENT_DIR / "*_oof.npz"))):
        data = np.load(path, allow_pickle=True)
        stem = os.path.basename(path).replace("_oof.npz", "")
        for key in data.files:
            arr = data[key]
            if getattr(arr, "shape", None) != y.shape or not np.isfinite(arr).all():
                continue
            if key in EXCLUDE_KEYS:
                print(f"  excluded (leaks the true tier): {stem}/{key}")
                continue
            preds.append(arr.astype(np.float64))
            names.append(f"{stem}/{key}"[:70])
    return np.array(preds).T, names


def blend_oof(P: np.ndarray, y: np.ndarray, folds: pd.DataFrame) -> np.ndarray:
    """Non-negative least squares weights, fitted per fold on the other folds."""
    out = np.zeros(len(y))
    for fold in sorted(folds["fold"].unique()):
        va = (folds["fold"] == fold).to_numpy()
        w, _ = nnls(P[~va], y[~va])
        out[va] = P[va] @ w
    return clip_prices(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=12,
                        help="candidate pool size, ranked by individual MSE")
    parser.add_argument("--out", default="e13_blend")
    args = parser.parse_args()

    folds = load_folds()
    y = folds["price"].to_numpy(dtype=np.float64)
    P, names = collect(y)
    singles = ((P - y[:, None]) ** 2).mean(axis=0)
    order = np.argsort(singles)
    print(f"\n{P.shape[1]} candidate predictions; best single MSE {singles.min():,.0f}")

    rows = []
    for k in (3, 5, 8, 12, 20, P.shape[1]):
        k = min(k, P.shape[1])
        idx = order[:k]
        oof = blend_oof(P[:, idx], y, folds)
        m = metrics(y, oof)
        m["n_models"] = k
        rows.append(m)
        print(f"  top-{k:<3} blend  MSE {m['mse']:>8,.0f}  R2 {m['r2']:.3f}  "
              f"slope {m['slope']:.3f}")
        if k == args.top:
            best_oof = oof

    results = pd.DataFrame(rows).sort_values("mse")
    results.to_csv(EXPERIMENT_DIR / f"{args.out}.csv", index=False)

    best_k = int(results.iloc[0]["n_models"])
    idx = order[:best_k]
    best_oof = blend_oof(P[:, idx], y, folds)
    np.savez(EXPERIMENT_DIR / f"{args.out}_oof.npz", blend=best_oof)

    # Weights refitted on everything, purely to report which models matter.
    w, _ = nnls(P[:, idx], y)
    print(f"\nbest: top-{best_k} blend, MSE {results.iloc[0]['mse']:,.0f} "
          f"(best single {singles.min():,.0f}, "
          f"{100 * (1 - results.iloc[0]['mse'] / singles.min()):+.2f}%)")
    print("\ncontributing models (weights fitted on all rows, for inspection only):")
    for i, wi in sorted(zip(idx, w), key=lambda t: -t[1]):
        if wi > 1e-3:
            print(f"  {wi:6.3f}  {names[i]:<70} (own MSE {singles[i]:,.0f})")

    print("\nper-band error of the blend:")
    print(band_errors(y, best_oof).to_string(index=False, float_format=lambda v: f"{v:,.1f}"))
    print("\nreliability (binned by prediction):")
    print(reliability(y, best_oof).to_string(index=False, float_format=lambda v: f"{v:,.1f}"))


if __name__ == "__main__":
    main()
