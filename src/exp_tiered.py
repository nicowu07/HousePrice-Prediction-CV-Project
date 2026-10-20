"""Does splitting the problem by price tier help?

The proposal: classify a house into low / mid / high from the photo, then
price it with a tier-specific regressor. This tests it, and separates the two
questions it bundles together.

*   **Can the tier be predicted at all?** If a three-way classifier on these
    features is barely better than chance, the idea is dead regardless of how
    good the experts are. Reported as classification accuracy and as the
    confusion between adjacent tiers.

*   **Would perfect routing even help?** Routing by the *true* tier gives an
    upper bound. It must be read carefully: the oracle leaks most of the
    answer, because knowing a house is in the top tier already narrows the
    price enormously. A large oracle gain is therefore expected and proves
    nothing on its own. Its use is to quantify what gate errors cost, by
    comparison with realistic routing.

*   **Does realisable routing help?** Hard routing uses the predicted tier.
    Soft routing weights each expert's prediction by the tier probability,
    which avoids committing to a wrong tier and degrades gracefully. Only
    these two are decision-relevant.

Prior from E4: a binned-distribution head -- effectively soft routing over
many tiers with constant experts -- did not beat a single regressor at any
bin count. A three-tier cascade is a coarser version of the same structure
with two extra liabilities: each expert sees roughly a third of 8,000 rows,
and a hard misroute cannot be recovered because the expert has never seen
prices outside its own tier.

Usage
-----
    python src/exp_tiered.py --seeds 0
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
    load_folds,
    metrics,
)
from exp_heads import MLP
from exp_objective import DEFAULT_TAGS, train_predict

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def tier_edges(prices: np.ndarray, scheme: str) -> np.ndarray:
    """Interior cut points. Tiers are [0, e0), [e0, e1), [e1, inf)."""
    if scheme == "tertile":
        return np.percentile(prices, [33.333, 66.667])
    if scheme == "fixed":
        # Mirrors how the project has been discussing the bands all along.
        return np.array([500.0, 1000.0])
    raise ValueError(scheme)


def assign_tier(prices: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.digitize(prices, edges)


def train_gate(Xtr, tiers, seed, n_tiers=3, epochs=40, bs=256, lr=1e-3):
    """Three-way tier classifier on the same features."""
    torch.manual_seed(seed)
    model = MLP(Xtr.shape[1], n_tiers).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    xb = torch.as_tensor(Xtr, dtype=torch.float32, device=DEVICE)
    yb = torch.as_tensor(tiers, dtype=torch.long, device=DEVICE)
    n = len(xb)
    steps = epochs * max(1, n // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.15)
    loss_fn = nn.CrossEntropyLoss()

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
def gate_probs(model, X):
    xb = torch.as_tensor(X, dtype=torch.float32, device=DEVICE)
    out = torch.cat([model(xb[i : i + 1024]).cpu() for i in range(0, len(xb), 1024)])
    return torch.softmax(out, dim=1).numpy()


def run_fold(X, y, tr, va, edges, seed):
    """Return a dict of raw-price predictions on the validation rows."""
    scaler = StandardScaler().fit(X[tr])
    Xtr, Xva = scaler.transform(X[tr]), scaler.transform(X[va])
    ytr, yva = y[tr], y[va]

    tiers_tr = assign_tier(ytr, edges)
    tiers_va = assign_tier(yva, edges)
    n_tiers = len(edges) + 1

    # --- single model reference, identical protocol to the baseline ---
    inner = np.zeros(tr.sum())
    for itr, iva in KFold(5, shuffle=True, random_state=seed).split(Xtr):
        inner[iva] = train_predict("A", Xtr[itr], ytr[itr], Xtr[iva], seed)
    cal = IsotonicRegression(out_of_bounds="clip").fit(inner, ytr)
    single = clip_prices(cal.predict(train_predict("A", Xtr, ytr, Xva, seed)))

    # --- tier experts ---
    expert_pred = np.zeros((len(yva), n_tiers))
    for t in range(n_tiers):
        mask = tiers_tr == t
        expert_pred[:, t] = train_predict("A", Xtr[mask], ytr[mask], Xva, seed)
    expert_pred = clip_prices(expert_pred)

    # --- gate ---
    gate = train_gate(Xtr, tiers_tr, seed, n_tiers)
    probs = gate_probs(gate, Xva)
    hard = probs.argmax(axis=1)

    return {
        "single": single,
        "oracle": expert_pred[np.arange(len(yva)), tiers_va],
        "hard": expert_pred[np.arange(len(yva)), hard],
        "soft": (probs * expert_pred).sum(axis=1),
        "gate_true": tiers_va,
        "gate_pred": hard,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tags", nargs="+", default=DEFAULT_TAGS)
    parser.add_argument("--scheme", choices=["tertile", "fixed"], default="tertile")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--out", default="e6_tiered")
    args = parser.parse_args()

    folds = load_folds()
    y = folds["price"].to_numpy(dtype=np.float64)
    X = np.concatenate([align_features(t, folds) for t in args.tags], axis=1)
    edges = tier_edges(y, args.scheme)
    tiers_all = assign_tier(y, edges)

    print(f"features {X.shape} | scheme {args.scheme} | cuts {np.round(edges, 1)}")
    print("tier sizes: " + ", ".join(
        f"tier {t} n={int((tiers_all == t).sum())} ({(tiers_all == t).mean() * 100:.1f}%)"
        for t in range(len(edges) + 1)))
    print(f"constant-prediction MSE = {y.var():,.0f}\n")

    keys = ["single", "oracle", "hard", "soft"]
    acc_all = []
    collected = {k: [] for k in keys}

    for seed in args.seeds:
        t0 = time.time()
        preds = {k: np.zeros(len(y)) for k in keys}
        gt, gp = np.zeros(len(y), int), np.zeros(len(y), int)
        for fold in sorted(folds["fold"].unique()):
            va = (folds["fold"] == fold).to_numpy()
            out = run_fold(X, y, ~va, va, edges, seed)
            for k in keys:
                preds[k][va] = out[k]
            gt[va], gp[va] = out["gate_true"], out["gate_pred"]
        for k in keys:
            collected[k].append(preds[k])
        acc_all.append((gt == gp).mean())
        print(f"seed {seed}: gate accuracy {(gt == gp).mean() * 100:.1f}%  [{time.time() - t0:.0f}s]")

    print(f"\ngate accuracy {np.mean(acc_all) * 100:.1f}%  "
          f"(chance {max(np.bincount(tiers_all)) / len(tiers_all) * 100:.1f}% by always "
          f"predicting the largest tier)")
    cm = pd.crosstab(pd.Series(gt, name="true"), pd.Series(gp, name="predicted"))
    print(cm.to_string())

    rows, store = [], {}
    for k in keys:
        oof = np.mean(collected[k], axis=0)
        store[k] = oof
        m = metrics(y, oof)
        m["routing"] = k
        rows.append(m)

    results = pd.DataFrame(rows)
    base = float(results.loc[results["routing"] == "single", "mse"].iloc[0])
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(EXPERIMENT_DIR / f"{args.out}.csv", index=False)
    np.savez(EXPERIMENT_DIR / f"{args.out}_oof.npz", **store)

    print("\n" + "=" * 70)
    print(f"{'routing':<10}{'MSE':>10}{'vs single':>11}{'MAE':>8}{'R2':>8}")
    for _, r in results.iterrows():
        print(f"{r['routing']:<10}{r['mse']:>10,.0f}{100 * (r['mse'] / base - 1):>10.2f}%"
              f"{r['mae']:>8.1f}{r['r2']:>8.3f}")
    print("\noracle leaks the true tier, so its gain is an upper bound, not a result.")
    print("The decision rests on 'hard' and 'soft' versus 'single'.")

    print("\nper-band error, single vs soft:")
    bs_, bf = band_errors(y, store["single"]), band_errors(y, store["soft"])
    cmp = bs_[["band", "n", "mse", "bias"]].merge(
        bf[["band", "mse", "bias"]], on="band", suffixes=("_single", "_soft"))
    cmp["mse_delta_%"] = 100 * (cmp["mse_soft"] / cmp["mse_single"] - 1)
    print(cmp.to_string(index=False, float_format=lambda v: f"{v:,.1f}"))


if __name__ == "__main__":
    main()
