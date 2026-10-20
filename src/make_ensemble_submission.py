"""Build the final ensemble: per-model OOF and test predictions, then blend.

Prediction-level fusion beat the best single model by ~5% on out-of-fold data
(see e13_blend), but that experiment could only use predictions we happened to
have stored, and those were out-of-fold only. To actually submit a blend we
need each member's test predictions too, so every member is refitted here:
nested cross-validation for its out-of-fold vector, and a full-training-set
fit for its test vector.

How the blend is kept honest:

*   **Mixing weights are non-negative least squares**, fitted per fold on the
    other four folds for the reported OOF score, and on all rows for the
    weights actually applied to the test predictions. Fitting weights and
    scoring on the same rows would be optimistic with this many members.
*   **A final linear recalibration** is fitted the same way. The raw blend
    comes out with slope ~1.07 -- a uniform tilt, because averaging several
    individually shrunk predictors compresses the spread further. A two
    parameter linear fix corrects exactly that and beat isotonic here (64,909
    vs 65,254), since isotonic spends its extra freedom on noise.

Usage
-----
    python src/make_ensemble_submission.py
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression

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
from exp_heads import CFG, HEADS

H336 = "vit_huge_plus_patch16_dinov3_336_squash"
H256 = "vit_huge_plus_patch16_dinov3_256_squash"
H224 = "vit_huge_plus_patch16_dinov3_224_squash"
L336 = "vit_large_patch16_dinov3_336_squash"
L224 = "vit_large_patch16_dinov3_224_squash"
TILE = "vit_large_patch16_dinov3_224_squash_tile2"
D2L336 = "vit_large_patch14_reg4_dinov2_336_squash"
D2L518 = "vit_large_patch14_reg4_dinov2_518_squash"
PM = "photometa"

# Chosen from the blend's fitted weights, keeping members that carry weight
# and are cheap to refit. Deliberately mixed in strength: the weak but
# decorrelated members earn their place by making different mistakes.
MEMBERS = [
    f"{H336}+{H256}+{L336}+{D2L336}+{TILE}+{PM}",
    f"{H336}+{L336}+{PM}",
    f"{H336}+{L336}+{TILE}+{PM}",
    f"{H224}+{H336}",
    H336,
    H256,
    L336,
    TILE,
    D2L518,
]


def oof_and_test(tag: str, y, folds, Xte_cache, head="mlp_mse", seed=0):
    parts = tag.split("+")
    X = np.concatenate([align_features(p, folds) for p in parts], axis=1)
    Xte = np.concatenate([Xte_cache[p] for p in parts], axis=1)
    fn = HEADS[head]
    CFG["seed"] = seed

    oof = np.zeros(len(y))
    for fold in sorted(folds["fold"].unique()):
        va = (folds["fold"] == fold).to_numpy()
        tr = ~va
        sc = StandardScaler().fit(X[tr])
        inner = np.zeros(tr.sum())
        Xtr = sc.transform(X[tr])
        for itr, iva in KFold(5, shuffle=True, random_state=seed).split(Xtr):
            inner[iva] = fn(Xtr[itr], y[tr][itr], Xtr[iva])
        cal = IsotonicRegression(out_of_bounds="clip").fit(inner, y[tr])
        oof[va] = clip_prices(cal.predict(fn(Xtr, y[tr], sc.transform(X[va]))))

    # Test vector: refit on all 8,000 rows, with the decoder still fitted on
    # out-of-fold predictions so it is not looking at its own training fit.
    sc = StandardScaler().fit(X)
    Xs, Xte_s = sc.transform(X), sc.transform(Xte)
    full_oof = np.zeros(len(y))
    for itr, iva in KFold(5, shuffle=True, random_state=seed).split(Xs):
        full_oof[iva] = fn(Xs[itr], y[itr], Xs[iva])
    cal = IsotonicRegression(out_of_bounds="clip").fit(full_oof, y)
    test = clip_prices(cal.predict(fn(Xs, y, Xte_s)))
    return oof, test


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--members", nargs="+", default=MEMBERS)
    parser.add_argument("--out", default="final_ensemble")
    args = parser.parse_args()

    folds = load_folds()
    y = folds["price"].to_numpy(dtype=np.float64)

    needed = sorted({p for m in args.members for p in m.split("+")})
    Xte_cache, test_ids = {}, None
    for p in needed:
        Xte_cache[p], ids = load_features(p, "test")
        test_ids = ids if test_ids is None else test_ids
    print(f"{len(args.members)} members | constant-prediction MSE {y.var():,.0f}\n")

    oofs, tests = [], []
    for i, tag in enumerate(args.members, 1):
        t0 = time.time()
        o, t = oof_and_test(tag, y, folds, Xte_cache)
        oofs.append(o)
        tests.append(t)
        short = tag.replace("vit_", "").replace("_patch16_dinov3", "_v3") \
                   .replace("_patch14_reg4_dinov2", "_v2").replace("_squash", "")
        print(f"[{i}/{len(args.members)}] {short[:64]:<64} MSE {((o - y) ** 2).mean():>8,.0f} "
              f"[{time.time() - t0:.0f}s]", flush=True)

    P = np.array(oofs).T
    T = np.array(tests).T

    # Honest OOF estimate: weights and recalibration both fitted out-of-fold.
    blended = np.zeros(len(y))
    for fold in sorted(folds["fold"].unique()):
        va = (folds["fold"] == fold).to_numpy()
        w, _ = nnls(P[~va], y[~va])
        raw_tr, raw_va = P[~va] @ w, P[va] @ w
        a, b = np.polyfit(raw_tr, y[~va], 1)
        blended[va] = a * raw_va + b
    blended = clip_prices(blended)

    m = metrics(y, blended)
    best_single = min(((o - y) ** 2).mean() for o in oofs)
    print(f"\nblend  MSE {m['mse']:,.0f}  R2 {m['r2']:.3f}  slope {m['slope']:.3f}")
    print(f"best single member {best_single:,.0f}  -> {100 * (1 - m['mse'] / best_single):+.2f}%")

    # Weights applied to the test set, fitted on all out-of-fold rows.
    w, _ = nnls(P, y)
    a, b = np.polyfit(P @ w, y, 1)
    test_pred = clip_prices(a * (T @ w) + b)
    print("\nweights:")
    for tag, wi in sorted(zip(args.members, w), key=lambda t: -t[1]):
        if wi > 1e-3:
            print(f"  {wi:6.3f}  {tag[:78]}")

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(EXPERIMENT_DIR / f"{args.out}_oof.npz", blend=blended,
             members=np.array(args.members, dtype=object))
    pd.DataFrame([m]).to_csv(EXPERIMENT_DIR / f"{args.out}.csv", index=False)

    print("\nper-band error:")
    print(band_errors(y, blended).to_string(index=False, float_format=lambda v: f"{v:,.1f}"))
    print("\nreliability (binned by prediction):")
    print(reliability(y, blended).to_string(index=False, float_format=lambda v: f"{v:,.1f}"))
    print()
    write_submission(test_ids, test_pred, args.out)


if __name__ == "__main__":
    main()
