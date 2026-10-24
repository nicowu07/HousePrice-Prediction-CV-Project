"""Export per-image out-of-fold predictions for error analysis.

Every row is a training image predicted by a model that never saw its label:
the fold column says which fold held it out. Nothing here is an in-sample
refit, so the errors are the ones the report should discuss.

Prices are in $1000 USD throughout, matching the competition.

Usage
-----
    python src/export_oof_predictions.py
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common import EXPERIMENT_DIR, PRICE_BANDS, load_features, load_folds

META_NAMES = [
    "width", "height", "aspect", "megapixels",
    "bytes", "bytes_per_pixel", "log_bytes",
    "sharpness", "sharpness_norm",
    "brightness_mean", "brightness_std",
    "saturation_mean", "saturation_std",
    "colorfulness",
    "sky_fraction", "green_fraction", "grey_fraction",
    "edge_density", "contrast_rms",
]
# Columns that help shortlist "odd viewpoint / occlusion" candidates without
# asserting anything about them; whoever picks the figures still looks.
KEEP_META = ["width", "height", "aspect", "sharpness_norm",
             "sky_fraction", "green_fraction", "edge_density"]


def band_of(price: float) -> str:
    for low, high in PRICE_BANDS:
        if low <= price < high:
            return f"{low}-{min(high, 2000)}"
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="final_ensemble_oof.npz")
    parser.add_argument("--key", default="blend")
    parser.add_argument("--model-name", default="nine-member NNLS ensemble (frozen features)")
    parser.add_argument("--out", default="oof_predictions_ensemble.csv")
    args = parser.parse_args()

    folds = load_folds()
    pred = np.load(EXPERIMENT_DIR / args.source, allow_pickle=True)[args.key]
    if pred.shape[0] != len(folds):
        raise SystemExit(f"length mismatch: {pred.shape[0]} predictions vs {len(folds)} rows")

    df = folds[["imageid", "price", "fold"]].copy()
    df["pred"] = pred
    df["error"] = df["pred"] - df["price"]          # positive = over-prediction
    df["abs_error"] = df["error"].abs()
    df["rel_error"] = df["error"] / df["price"]
    df["sq_error"] = df["error"] ** 2
    df["band"] = df["price"].apply(band_of)
    df["model"] = args.model_name

    # Rank within the whole set, so a case can be described as "the Nth worst
    # over-prediction" rather than just "a bad one".
    df["abs_error_rank"] = df["abs_error"].rank(ascending=False).astype(int)
    df["pct_abs_error"] = df["abs_error"].rank(pct=True) * 100

    # Attach the low-level photographic statistics already computed in E7.
    # These are descriptive only -- they shortlist candidates for a visual
    # check, they do not explain anything on their own.
    try:
        meta, meta_ids = load_features("photometa", "train")
        order = {img: i for i, img in enumerate(meta_ids)}
        idx = np.array([order[i] for i in df["imageid"]])
        for name in KEEP_META:
            df[name] = meta[idx, META_NAMES.index(name)]
    except SystemExit:
        print("note: photometa features not found, skipping descriptive columns")

    cols = ["imageid", "price", "pred", "error", "abs_error", "rel_error",
            "sq_error", "band", "fold", "abs_error_rank", "pct_abs_error",
            "model"] + [c for c in KEEP_META if c in df.columns]
    df = df[cols].sort_values("imageid", key=lambda s: s.str.replace(".jpg", "", regex=False).astype(int))

    out = EXPERIMENT_DIR / args.out
    df.to_csv(out, index=False, float_format="%.3f")
    print(f"wrote {out}  ({len(df)} rows)")
    print(f"model: {args.model_name}")
    print(f"overall MSE {df['sq_error'].mean():,.0f}  MAE {df['abs_error'].mean():.1f}  "
          f"R2 {1 - df['sq_error'].mean() / df['price'].var():.3f}")
    print("\nper-fold check (each fold held out by its own model):")
    print(df.groupby("fold").agg(n=("price", "size"), mse=("sq_error", "mean"),
                                 mae=("abs_error", "mean")).to_string(
        float_format=lambda v: f"{v:,.1f}"))


if __name__ == "__main__":
    main()
