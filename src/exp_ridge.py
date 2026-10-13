"""E1: target scale and decoding, plus the aspect-ratio treatment.

Two controlled comparisons on frozen features with a Ridge regressor, so that
nothing but the factor under test changes:

1.  **Aspect-ratio treatment** (squash / center / pad) -- how the 4:3 listing
    photo is made square before it reaches the backbone.

2.  **Target scale and decoding** -- the metric is MSE on raw price, whose
    Bayes-optimal predictor is E[price|x]. Fitting on raw price is unstable
    because the target is right-skewed; fitting on log price is stable but
    exp() of the fitted value estimates the conditional *median*, which is
    systematically low. Each decoder is a different way of recovering the
    conditional mean.

Nested cross-validation: the Ridge penalty and the decoder are both fitted
inside the training portion of each outer fold, using inner out-of-fold
predictions. In-sample residuals are too small, so a smearing factor or
residual variance estimated from them would under-correct and the outer score
would be optimistic.

Usage
-----
    python src/exp_ridge.py
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from common import (
    EXPERIMENT_DIR,
    TargetDecoder,
    align_features,
    band_errors,
    format_metrics,
    load_features,
    load_folds,
    metrics,
    reliability,
    write_submission,
)

ALPHAS = np.logspace(0, 6, 25)
DECODERS = {
    "raw": "raw",
    "log_naive": "log",
    "log_smearing": "log",
    "log_lognormal": "log",
    "log_isotonic": "log",
}


def fit_space(X: np.ndarray, y: np.ndarray, folds: pd.DataFrame, space: str, seed: int = 0):
    """Nested-CV Ridge in one target space.

    Returns the outer out-of-fold predictions (in that space) and, per outer
    fold, the inner out-of-fold predictions needed to fit a decoder honestly.
    """
    target = y if space == "raw" else np.log(y)
    oof = np.zeros(len(y))
    inner = {}  # fold -> (inner_oof_pred, y_train)
    alphas_used = []

    for fold in sorted(folds["fold"].unique()):
        va = (folds["fold"] == fold).to_numpy()
        tr = ~va

        scaler = StandardScaler().fit(X[tr])
        Xtr, Xva = scaler.transform(X[tr]), scaler.transform(X[va])
        ttr = target[tr]

        # Efficient leave-one-out generalised CV picks the penalty without a
        # separate loop over alphas.
        model = RidgeCV(alphas=ALPHAS).fit(Xtr, ttr)
        alphas_used.append(float(model.alpha_))
        oof[va] = model.predict(Xva)

        inner_oof = np.zeros(tr.sum())
        for itr, iva in KFold(5, shuffle=True, random_state=seed).split(Xtr):
            m = Ridge(alpha=model.alpha_).fit(Xtr[itr], ttr[itr])
            inner_oof[iva] = m.predict(Xtr[iva])
        inner[fold] = (inner_oof, y[tr])

    return oof, inner, alphas_used


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tags", nargs="+", default=[
        "vit_base_patch14_reg4_dinov2_224_squash",
        "vit_base_patch14_reg4_dinov2_224_center",
        "vit_base_patch14_reg4_dinov2_224_pad",
    ])
    parser.add_argument("--decoders", nargs="+", default=list(DECODERS),
                        help="subset of decoders to evaluate")
    parser.add_argument("--write-submission", action="store_true",
                        help="refit the best configuration on all training data and WRITE a Kaggle-format CSV locally (does not upload)")
    parser.add_argument("--out", default="e1_ridge_target_scale")
    args = parser.parse_args()

    decoders = {k: v for k, v in DECODERS.items() if k in args.decoders}
    if not decoders:
        raise SystemExit(f"no valid decoders in {args.decoders}; choose from {list(DECODERS)}")

    folds = load_folds()
    y = folds["price"].to_numpy(dtype=np.float64)
    print(f"{len(folds)} training rows | constant-prediction MSE = {y.var():,.0f}\n")

    rows = []
    store = {}

    for tag in args.tags:
        X = align_features(tag, folds)
        print(f"--- {tag}  features {X.shape} ---")
        spaces = {}
        for space in sorted(set(decoders.values())):
            spaces[space] = fit_space(X, y, folds, space)

        for mode, space in decoders.items():
            oof_space, inner, alphas = spaces[space]
            pred = np.zeros(len(y))
            for fold in sorted(folds["fold"].unique()):
                va = (folds["fold"] == fold).to_numpy()
                inner_oof, y_tr = inner[fold]
                decoder = TargetDecoder(mode).fit(inner_oof, y_tr)
                pred[va] = decoder.decode(oof_space[va])

            m = metrics(y, pred)
            m.update(tag=tag, decode=mode, alpha_median=float(np.median(alphas)))
            rows.append(m)
            store[(tag, mode)] = pred
            print("  " + format_metrics(mode, m))
        print()

    results = pd.DataFrame(rows).sort_values("mse").reset_index(drop=True)
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(EXPERIMENT_DIR / f"{args.out}.csv", index=False)

    print("=" * 96)
    print("RANKED BY OUT-OF-FOLD MSE")
    print("=" * 96)
    cols = ["tag", "decode", "mse", "rmse", "mae", "r2", "bias", "slope"]
    show = results[cols].copy()
    show["tag"] = show["tag"].str.replace("vit_base_patch14_reg4_dinov2_224_", "", regex=False)
    print(show.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))

    best = results.iloc[0]
    print(f"\nbest: {best['tag']} / {best['decode']}")
    print(f"  MSE {best['mse']:,.0f} vs baseline {y.var():,.0f} "
          f"({100 * (1 - best['mse'] / y.var()):.1f}% of the baseline error removed)")

    print("\nper-band error for the best configuration:")
    bands = band_errors(y, store[(best["tag"], best["decode"])])
    print(bands.to_string(index=False, float_format=lambda v: f"{v:,.1f}"))
    bands.to_csv(EXPERIMENT_DIR / f"{args.out}_bands.csv", index=False)

    print("\nreliability (binned by PREDICTION - the honest calibration test):")
    rel = reliability(y, store[(best["tag"], best["decode"])])
    print(rel.to_string(index=False, float_format=lambda v: f"{v:,.1f}"))
    rel.to_csv(EXPERIMENT_DIR / f"{args.out}_reliability.csv", index=False)

    np.save(EXPERIMENT_DIR / f"{args.out}_oof.npy", store[(best["tag"], best["decode"])])
    (EXPERIMENT_DIR / f"{args.out}_config.json").write_text(
        json.dumps({"tag": best["tag"], "decode": best["decode"], "mse": best["mse"]}, indent=2))

    if args.write_submission:
        tag, mode = best["tag"], best["decode"]
        space = decoders[mode]
        X = align_features(tag, folds)
        Xte, test_ids = load_features(tag, "test")
        target = y if space == "raw" else np.log(y)

        scaler = StandardScaler().fit(X)
        Xs, Xte_s = scaler.transform(X), scaler.transform(Xte)
        model = RidgeCV(alphas=ALPHAS).fit(Xs, target)

        # Decoder fitted on out-of-fold predictions over the full training set.
        full_oof = np.zeros(len(y))
        for itr, iva in KFold(5, shuffle=True, random_state=0).split(Xs):
            m = Ridge(alpha=model.alpha_).fit(Xs[itr], target[itr])
            full_oof[iva] = m.predict(Xs[iva])
        decoder = TargetDecoder(mode).fit(full_oof, y)

        preds = decoder.decode(model.predict(Xte_s))
        print()
        write_submission(test_ids, preds, f"ridge_{tag}_{mode}")


if __name__ == "__main__":
    main()
