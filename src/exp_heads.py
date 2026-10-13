"""E3/E4: which features, and which head?

Heads, all on identical inputs under an identical nested-CV protocol, so the
only thing that varies is how the mapping from features to price is modelled:

*   **Ridge** -- linear reference from E1.
*   **LightGBM** -- axis-aligned non-linearity. Dense ViT embeddings have no
    individually meaningful dimensions, so tree splits are a poor match; this
    tests that expectation rather than assuming it.
*   **MLP + MSE** -- non-linear point regressor on log price.
*   **MLP + binned distribution** -- predicts a distribution over log-spaced
    price bins with soft-target cross-entropy and decodes the expectation.
*   **MLP + hybrid** -- the same distributional head with an added squared
    error term on the decoded mean.

The last three share a trunk and differ only in output layer and loss, which
is what makes the comparison attributable to the objective rather than to
capacity.

The hybrid head exists because of what the pure distributional head showed:
cross-entropy is a proper scoring rule for the whole predicted distribution,
i.e. it minimises KL to the smoothed label distribution. That is *not* the
same as minimising the squared error of the distribution's mean, which is
what the competition scores. Where signal is strong the two objectives point
the same way, but with R^2 around 0.45 and 6,400 training rows per fold, the
extra estimation variance of fitting 64 outputs instead of 1 can cost more
than the bounded-gradient robustness gains. The hybrid term reintroduces the
metric directly while keeping the distributional representation (and with it
the predictive uncertainty used in error analysis).

Usage
-----
    python src/exp_heads.py --tags <tag> [<tag> ...] --heads mlp_mse ridge
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from bins import PriceBinner
from common import (
    load_features,
    write_submission,
    EXPERIMENT_DIR,
    align_features,
    band_errors,
    clip_prices,
    format_metrics,
    load_folds,
    metrics,
    reliability,
)

ALPHAS = np.logspace(0, 6, 25)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Set from the CLI so the binned heads can be swept without editing code.
CFG = {"n_bins": 64, "smoothing": 1.5, "lambda_mse": 1.0, "epochs": 40, "seed": 0}


# --------------------------------------------------------------------------
# heads: each maps (train features, train prices, eval features) -> prices
# --------------------------------------------------------------------------

def head_ridge(Xtr, ytr, Xev):
    return np.exp(RidgeCV(alphas=ALPHAS).fit(Xtr, np.log(ytr)).predict(Xev))


def head_lightgbm(Xtr, ytr, Xev):
    import lightgbm as lgb

    model = lgb.LGBMRegressor(
        n_estimators=400, learning_rate=0.06, num_leaves=31,
        feature_fraction=0.1, bagging_fraction=0.8, bagging_freq=1,
        min_child_samples=40, n_jobs=-1, verbose=-1, random_state=0,
    ).fit(Xtr, np.log(ytr))
    return np.exp(model.predict(Xev))


class MLP(nn.Module):
    """Shared trunk; output width is what distinguishes the MLP heads."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 512, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.BatchNorm1d(hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def _train_mlp(Xtr, targets, out_dim, loss_fn, epochs=None, bs=256, lr=1e-3, seed=None):
    epochs = CFG["epochs"] if epochs is None else epochs
    torch.manual_seed(CFG["seed"] if seed is None else seed)
    model = MLP(Xtr.shape[1], out_dim).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    xb = torch.as_tensor(Xtr, dtype=torch.float32, device=DEVICE)
    yb = torch.as_tensor(targets, dtype=torch.float32, device=DEVICE)
    n = len(xb)
    steps = epochs * max(1, n // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.15)

    model.train()
    step = 0
    for _ in range(epochs):
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n - bs + 1, bs):
            idx = perm[i : i + bs]
            loss = loss_fn(model(xb[idx]), yb[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step < steps - 1:
                sched.step()
            step += 1
    model.eval()
    return model


@torch.no_grad()
def _predict(model, X):
    xb = torch.as_tensor(X, dtype=torch.float32, device=DEVICE)
    return torch.cat([model(xb[i : i + 1024]).cpu() for i in range(0, len(xb), 1024)])


def head_mlp_mse(Xtr, ytr, Xev):
    t = np.log(ytr)
    mu, sd = t.mean(), t.std()
    model = _train_mlp(Xtr, ((t - mu) / sd)[:, None], 1, nn.MSELoss())
    return np.exp(_predict(model, Xev).numpy().ravel() * sd + mu)


def _binned_head(Xtr, ytr, Xev, lambda_mse: float):
    binner = PriceBinner(n_bins=CFG["n_bins"], smoothing=CFG["smoothing"]).fit(ytr)
    # Last column carries the true price so the hybrid loss can reach it.
    targets = np.concatenate([binner.soft_targets(ytr), ytr[:, None].astype(np.float32)], axis=1)
    centers = torch.as_tensor(binner.centers, dtype=torch.float32, device=DEVICE)
    k = binner.n_bins
    scale = 1000.0  # keep the price term O(1) so lambda is interpretable

    def loss_fn(logits, target):
        soft, price = target[:, :k], target[:, k]
        ce = -(soft * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
        if lambda_mse == 0.0:
            return ce
        mean = (torch.softmax(logits, dim=1) * centers).sum(dim=1)
        return ce + lambda_mse * ((mean - price) / scale).pow(2).mean()

    model = _train_mlp(Xtr, targets, k, loss_fn)
    probs = torch.softmax(_predict(model, Xev), dim=1).numpy()
    return binner.decode_mean(probs)


def head_mlp_binned(Xtr, ytr, Xev):
    return _binned_head(Xtr, ytr, Xev, lambda_mse=0.0)


def head_mlp_hybrid(Xtr, ytr, Xev):
    return _binned_head(Xtr, ytr, Xev, lambda_mse=CFG["lambda_mse"])


HEADS = {
    "ridge": head_ridge,
    "lightgbm": head_lightgbm,
    "mlp_mse": head_mlp_mse,
    "mlp_binned": head_mlp_binned,
    "mlp_hybrid": head_mlp_hybrid,
}


# --------------------------------------------------------------------------

def evaluate_head(fn, X, y, folds, seed=0):
    """Nested CV; inner out-of-fold predictions fit the isotonic calibrator."""
    oof = np.zeros(len(y))
    for fold in sorted(folds["fold"].unique()):
        va = (folds["fold"] == fold).to_numpy()
        tr = ~va
        scaler = StandardScaler().fit(X[tr])
        Xtr, Xva = scaler.transform(X[tr]), scaler.transform(X[va])
        ytr = y[tr]

        inner = np.zeros(tr.sum())
        for itr, iva in KFold(5, shuffle=True, random_state=seed).split(Xtr):
            inner[iva] = fn(Xtr[itr], ytr[itr], Xtr[iva])
        calibrator = IsotonicRegression(out_of_bounds="clip").fit(inner, ytr)
        oof[va] = clip_prices(calibrator.predict(fn(Xtr, ytr, Xva)))
    return oof


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tags", nargs="+", default=["vit_base_patch14_reg4_dinov2_224_squash"])
    parser.add_argument("--heads", nargs="+", default=["ridge", "mlp_mse"])
    parser.add_argument("--bins", type=int, default=64)
    parser.add_argument("--smoothing", type=float, default=1.5)
    parser.add_argument("--lambda-mse", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0],
                        help="repeat each run with these seeds to gauge run-to-run stability")
    parser.add_argument("--out", default="e3_e4_sweep")
    parser.add_argument("--write-submission", action="store_true",
                        help="refit the best configuration on all training data and WRITE a "
                             "Kaggle-format CSV locally (does not upload)")
    args = parser.parse_args()

    CFG.update(n_bins=args.bins, smoothing=args.smoothing,
               lambda_mse=args.lambda_mse, epochs=args.epochs)

    folds = load_folds()
    y = folds["price"].to_numpy(dtype=np.float64)
    print(f"device {DEVICE} | constant-prediction MSE = {y.var():,.0f}\n")

    rows, store = [], {}
    for tag in args.tags:
        # A '+' joins several cached extractions into one feature vector.
        # Concatenating before the head lets it learn interactions between the
        # backbones, which averaging their separate predictions cannot do.
        # No scaling here: evaluate_head fits a StandardScaler per fold, which
        # already puts every dimension on a common scale without leaking
        # validation statistics into the transform.
        parts = tag.split("+")
        X = np.concatenate([align_features(p, folds) for p in parts], axis=1)
        short = tag.replace("vit_", "").replace("_patch14_reg4_dinov2", "").replace("convnext_", "cnx_")
        for name in args.heads:
            per_seed = []
            t0 = time.time()
            for seed in args.seeds:
                CFG["seed"] = seed
                oof = evaluate_head(HEADS[name], X, y, folds, seed=seed)
                per_seed.append(oof)
            oof = np.mean(per_seed, axis=0)  # seed-averaged predictions
            m = metrics(y, oof)
            m.update(tag=tag, head=name, dim=X.shape[1], seconds=time.time() - t0,
                     mse_seed_std=float(np.std([metrics(y, p)["mse"] for p in per_seed])))
            rows.append(m)
            store[f"{tag}|{name}"] = oof
            label = f"{short} / {name}"
            extra = f"  +-{m['mse_seed_std']:,.0f}" if len(args.seeds) > 1 else ""
            print(format_metrics(label, m) + f"{extra}  [{m['seconds']:.0f}s]")

    results = pd.DataFrame(rows).sort_values("mse").reset_index(drop=True)
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(EXPERIMENT_DIR / f"{args.out}.csv", index=False)
    np.savez(EXPERIMENT_DIR / f"{args.out}_oof.npz", **store)

    print("\n" + "=" * 96)
    show = results[["tag", "head", "dim", "mse", "rmse", "mae", "r2", "slope"]].copy()
    show["tag"] = show["tag"].str.replace("vit_", "", regex=False).str.replace("_patch14_reg4_dinov2", "", regex=False)
    print(show.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))

    best = results.iloc[0]
    key = f"{best['tag']}|{best['head']}"
    print(f"\nbest: {best['tag']} / {best['head']}  MSE {best['mse']:,.0f} "
          f"({100 * (1 - best['mse'] / y.var()):.1f}% of baseline error removed)")
    print("\nper-band error (binned by TRUE price):")
    print(band_errors(y, store[key]).to_string(index=False, float_format=lambda v: f"{v:,.1f}"))
    print("\nreliability (binned by PREDICTION):")
    print(reliability(y, store[key]).to_string(index=False, float_format=lambda v: f"{v:,.1f}"))

    if args.write_submission:
        tag, head = best["tag"], best["head"]
        parts = tag.split("+")
        X = np.concatenate([align_features(p, folds) for p in parts], axis=1)
        Xte = np.concatenate([load_features(p, "test")[0] for p in parts], axis=1)
        test_ids = load_features(parts[0], "test")[1]

        # Refit on all 8,000 training rows. The isotonic decoder is still
        # fitted on out-of-fold predictions only -- in-sample predictions
        # would make it close to the identity and undo the calibration.
        scaler = StandardScaler().fit(X)
        Xs, Xte_s = scaler.transform(X), scaler.transform(Xte)

        preds = []
        for seed in args.seeds:
            CFG["seed"] = seed
            oof = np.zeros(len(y))
            for itr, iva in KFold(5, shuffle=True, random_state=seed).split(Xs):
                oof[iva] = HEADS[head](Xs[itr], y[itr], Xs[iva])
            calibrator = IsotonicRegression(out_of_bounds="clip").fit(oof, y)
            preds.append(calibrator.predict(HEADS[head](Xs, y, Xte_s)))
        print()
        write_submission(test_ids, np.mean(preds, axis=0), f"{args.out}_{head}")


if __name__ == "__main__":
    main()
