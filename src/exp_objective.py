"""Training objective study: target scale and price-dependent sample weighting.

The competition scores MSE on raw price, but the baseline MLP is trained on
squared error in log price. Those are not the same objective, and the gap has
a closed form. Writing the prediction as p_hat = exp(mu_hat) and the target as
p = exp(z),

    (p_hat - p)^2 = e^{2z} (e^{mu_hat - z} - 1)^2  ~=  p^2 (mu_hat - z)^2

to first order in the residual. So **raw-price MSE is log-space MSE weighted
by p^2**. Training unweighted in log space therefore under-weights expensive
houses by exactly the factor the metric cares about: at equal relative error,
a $2M house contributes 25x the squared error of a $400k house.

This reframes two separate proposals as one family. Switching the target to
raw price, and up-weighting expensive houses, are both corrections for the
same mismatch. The principled weight is w ∝ p^2, and a heuristic "2x above
$1M" sits somewhere between p^0 and p^1. So instead of three disconnected
conditions we sweep one parameter:

    w ∝ p^alpha ,  alpha = 0 (current baseline) ... 2 (first-order raw MSE)

with the raw-price target and the 2x heuristic included as reference points.
Theory predicts the optimum near alpha = 2; the first-order expansion assumes
small residuals and ours are not small, so a different optimum is itself an
informative result rather than a failure.

Two protocol choices worth stating:

*   **Loss weighting, not resampling.** They have the same expected gradient,
    but resampling adds sampling noise and changes the price composition of
    each batch, which perturbs the BatchNorm statistics. That would introduce
    a second variable into a comparison meant to isolate one.
*   **Weights are normalised to mean 1** within each training split, so the
    total gradient scale -- and with it the effective learning rate -- is
    comparable across conditions. Without this, "weighting" and "higher
    learning rate" would be confounded.

Weights are applied to training only. Validation keeps the natural price
distribution, and the isotonic calibrator is fitted on unweighted inner
out-of-fold predictions, exactly as in the baseline.

Usage
-----
    python src/exp_objective.py --seeds 0            # trace the curve
    python src/exp_objective.py --seeds 0 1 2 --conditions A alpha2.0
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from common import (
    EXPERIMENT_DIR,
    align_features,
    band_errors,
    clip_prices,
    load_features,
    load_folds,
    metrics,
    reliability,
    write_submission,
)
from exp_heads import MLP

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_TAGS = [
    "vit_large_patch14_reg4_dinov2_224_squash",
    "vit_large_patch14_reg4_dinov2_336_squash",
    "convnext_base_288_squash",
    "vit_base_patch14_reg4_dinov2_336_squash",
]


# --------------------------------------------------------------------------
# conditions
# --------------------------------------------------------------------------

def weights_for(condition: str, prices: np.ndarray) -> np.ndarray | None:
    """Per-sample training weight, normalised to mean 1. None = unweighted."""
    if condition in ("A", "B"):
        return None
    if condition == "C":
        # Hongze's heuristic: expensive houses count double.
        w = np.where(prices >= 1000, 2.0, 1.0)
    elif condition.startswith("alpha"):
        alpha = float(condition[5:])
        w = prices ** alpha
    else:
        raise ValueError(f"unknown condition {condition!r}")
    return (w / w.mean()).astype(np.float32)


CONDITIONS = {
    "A":         "log target, unweighted (current baseline)",
    "B":         "raw-price target, standardised",
    "C":         "log target, 2x weight for price >= 1000",
    "alpha0.5":  "log target, w ~ p^0.5",
    "alpha1.0":  "log target, w ~ p^1",
    "alpha1.5":  "log target, w ~ p^1.5",
    "alpha2.0":  "log target, w ~ p^2  (first-order equivalent of raw MSE)",
    "alpha3.0":  "log target, w ~ p^3  (past the predicted optimum)",
}


# --------------------------------------------------------------------------

def train_predict(condition, Xtr, ytr, Xev, seed, epochs=40, bs=256, lr=1e-3):
    """Fit one head under one condition; return raw-price predictions."""
    raw_target = condition == "B"
    if raw_target:
        t = ytr.astype(np.float64)
    else:
        t = np.log(ytr)
    mu, sd = t.mean(), t.std()
    t_std = ((t - mu) / sd).astype(np.float32)

    w = weights_for(condition, ytr)

    torch.manual_seed(seed)
    model = MLP(Xtr.shape[1], 1).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    xb = torch.as_tensor(Xtr, dtype=torch.float32, device=DEVICE)
    yb = torch.as_tensor(t_std, dtype=torch.float32, device=DEVICE)
    wb = None if w is None else torch.as_tensor(w, dtype=torch.float32, device=DEVICE)

    n = len(xb)
    steps = epochs * max(1, n // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.15)

    model.train()
    step = 0
    for _ in range(epochs):
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n - bs + 1, bs):
            idx = perm[i : i + bs]
            pred = model(xb[idx]).squeeze(1)
            sq = (pred - yb[idx]) ** 2
            # Normalise by the batch's own weight sum so that the gradient
            # scale does not drift between batches that happen to contain
            # more or fewer expensive houses.
            loss = sq.mean() if wb is None else (wb[idx] * sq).sum() / wb[idx].sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step < steps - 1:
                sched.step()
            step += 1

    model.eval()
    with torch.no_grad():
        xe = torch.as_tensor(Xev, dtype=torch.float32, device=DEVICE)
        out = torch.cat([model(xe[i : i + 1024]).cpu() for i in range(0, len(xe), 1024)])
    pred = out.numpy().ravel() * sd + mu
    return pred if raw_target else np.exp(pred)


def evaluate(condition, X, y, folds, seed):
    """Nested CV, identical to the baseline protocol."""
    oof = np.zeros(len(y))
    for fold in sorted(folds["fold"].unique()):
        va = (folds["fold"] == fold).to_numpy()
        tr = ~va
        scaler = StandardScaler().fit(X[tr])
        Xtr, Xva = scaler.transform(X[tr]), scaler.transform(X[va])
        ytr = y[tr]

        inner = np.zeros(tr.sum())
        for itr, iva in KFold(5, shuffle=True, random_state=seed).split(Xtr):
            inner[iva] = train_predict(condition, Xtr[itr], ytr[itr], Xtr[iva], seed)
        calibrator = IsotonicRegression(out_of_bounds="clip").fit(inner, ytr)
        oof[va] = clip_prices(calibrator.predict(train_predict(condition, Xtr, ytr, Xva, seed)))
    return oof


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tags", nargs="+", default=DEFAULT_TAGS)
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--out", default="e5_objective")
    parser.add_argument("--write-submission", action="store_true",
                        help="refit the best condition on all training data and WRITE a "
                             "Kaggle-format CSV locally (does not upload)")
    args = parser.parse_args()

    folds = load_folds()
    y = folds["price"].to_numpy(dtype=np.float64)
    X = np.concatenate([align_features(t, folds) for t in args.tags], axis=1)
    print(f"features {X.shape} | seeds {args.seeds} | device {DEVICE}")
    print(f"constant-prediction MSE = {y.var():,.0f}")
    print(f"share of rows with price >= 1000: {(y >= 1000).mean() * 100:.1f}%\n")

    rows, store = [], {}
    for cond in args.conditions:
        t0 = time.time()
        per_seed = [evaluate(cond, X, y, folds, s) for s in args.seeds]
        oof = np.mean(per_seed, axis=0)
        m = metrics(y, oof)
        seed_mses = [metrics(y, p)["mse"] for p in per_seed]
        m.update(condition=cond, description=CONDITIONS.get(cond, ""),
                 seconds=time.time() - t0, mse_seed_std=float(np.std(seed_mses)),
                 mse_seed_mean=float(np.mean(seed_mses)))
        rows.append(m)
        store[cond] = oof

        w = weights_for(cond, y)
        wr = "" if w is None else f"  w range {w.min():.3f}-{w.max():.2f}"
        extra = f" +-{m['mse_seed_std']:,.0f}" if len(args.seeds) > 1 else ""
        print(f"{cond:<9} MSE {m['mse']:>8,.0f}{extra}  MAE {m['mae']:>6.1f}  "
              f"R2 {m['r2']:.3f}  slope {m['slope']:.3f}  [{m['seconds']:.0f}s]{wr}")

    results = pd.DataFrame(rows).sort_values("mse").reset_index(drop=True)
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(EXPERIMENT_DIR / f"{args.out}.csv", index=False)
    np.savez(EXPERIMENT_DIR / f"{args.out}_oof.npz", **store)

    base = results.loc[results["condition"] == "A", "mse"]
    base = float(base.iloc[0]) if len(base) else float("nan")
    print("\n" + "=" * 92)
    print(f"{'condition':<10}{'MSE':>10}{'vs A':>9}{'MAE':>8}{'R2':>7}{'slope':>7}   description")
    for _, r in results.iterrows():
        delta = f"{100 * (r['mse'] / base - 1):+.2f}%" if base == base else "n/a"
        print(f"{r['condition']:<10}{r['mse']:>10,.0f}{delta:>9}{r['mae']:>8.1f}"
              f"{r['r2']:>7.3f}{r['slope']:>7.3f}   {r['description']}")

    best = results.iloc[0]
    print(f"\nper-band error, A (baseline) vs {best['condition']} (best):")
    ba, bb = band_errors(y, store["A"]), band_errors(y, store[best["condition"]])
    cmp = ba[["band", "n", "mse", "bias"]].merge(
        bb[["band", "mse", "bias"]], on="band", suffixes=("_A", "_best"))
    cmp["mse_delta_%"] = 100 * (cmp["mse_best"] / cmp["mse_A"] - 1)
    print(cmp.to_string(index=False, float_format=lambda v: f"{v:,.1f}"))

    print(f"\nreliability of {best['condition']} (binned by PREDICTION):")
    print(reliability(y, store[best["condition"]]).to_string(
        index=False, float_format=lambda v: f"{v:,.1f}"))

    if args.write_submission:
        cond = best["condition"]
        Xte = np.concatenate([load_features(t, "test")[0] for t in args.tags], axis=1)
        test_ids = load_features(args.tags[0], "test")[1]
        scaler = StandardScaler().fit(X)
        Xs, Xte_s = scaler.transform(X), scaler.transform(Xte)

        preds = []
        for seed in args.seeds:
            oof = np.zeros(len(y))
            for itr, iva in KFold(5, shuffle=True, random_state=seed).split(Xs):
                oof[iva] = train_predict(cond, Xs[itr], y[itr], Xs[iva], seed)
            cal = IsotonicRegression(out_of_bounds="clip").fit(oof, y)
            preds.append(cal.predict(train_predict(cond, Xs, y, Xte_s, seed)))
        print()
        write_submission(test_ids, np.mean(preds, axis=0), f"{args.out}_{cond}")


if __name__ == "__main__":
    main()
