"""Low-level photographic metadata as a feature block.

Every semantic feature we extract is computed after the image has been resized
to a square at a fixed resolution, which throws away three things the file
itself carries: its original dimensions, its aspect ratio, and how many bytes
the JPEG encoder needed. Those are not nothing. Listing photography is a paid
service, and higher-value listings are more likely to be shot professionally:
wider lens, cleaner exposure, sharper focus, blue sky, a larger and less
compressed file. If that holds, these cheap statistics carry price signal that
the backbone never sees.

Two readings are possible and the experiment is worth running either way. If
they help, the honest framing is that part of the signal is a property of the
listing rather than of the house -- a shortcut, which belongs in the error
analysis and in the limitations. If they do not help, that is evidence the
model is reading the house itself.

All features are derived from the provided images only, so nothing here steps
outside what the brief permits.

Usage
-----
    python src/extract_meta.py --split train
    python src/extract_meta.py --split test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "house_dataset"
FEATURE_DIR = PROJECT_ROOT / "features"

NAMES = [
    "width", "height", "aspect", "megapixels",
    "bytes", "bytes_per_pixel", "log_bytes",
    "sharpness", "sharpness_norm",
    "brightness_mean", "brightness_std",
    "saturation_mean", "saturation_std",
    "colorfulness",
    "sky_fraction", "green_fraction", "grey_fraction",
    "edge_density", "contrast_rms",
]


def describe(path: Path) -> np.ndarray:
    n_bytes = path.stat().st_size
    with Image.open(path) as im:
        w, h = im.size
        rgb = np.asarray(im.convert("RGB"))

    bgr = rgb[:, :, ::-1]
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0].astype(np.float32), hsv[..., 1] / 255.0, hsv[..., 2] / 255.0

    # Variance of the Laplacian is the standard no-reference focus measure.
    # It scales with resolution, so keep both the raw and the per-pixel form.
    lap = cv2.Laplacian(grey, cv2.CV_64F).var()

    # Hasler & Suesstrunk's colourfulness: professional work tends to be
    # more saturated and more colour-balanced than a phone snapshot.
    r, g, b = rgb[..., 0].astype(np.float32), rgb[..., 1].astype(np.float32), rgb[..., 2].astype(np.float32)
    rg, yb = r - g, 0.5 * (r + g) - b
    colourfulness = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2)
                          + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))

    # Sky is measured in the top third only, where it actually is: a clear
    # blue sky is both a weather/market signal and a sign of a planned shoot.
    top = slice(0, max(1, rgb.shape[0] // 3))
    sky = float((((hue[top] > 90) & (hue[top] < 135)) & (sat[top] > 0.25) & (val[top] > 0.4)).mean())
    green = float((((hue > 25) & (hue < 85)) & (sat > 0.2) & (val > 0.15)).mean())
    grey_frac = float((sat < 0.12).mean())

    edges = cv2.Canny(grey, 80, 200)
    return np.array([
        w, h, w / h, w * h / 1e6,
        n_bytes, n_bytes / (w * h), np.log(n_bytes),
        lap, lap / (w * h) * 1e4,
        float(val.mean()), float(val.std()),
        float(sat.mean()), float(sat.std()),
        colourfulness,
        sky, green, grey_frac,
        float((edges > 0).mean()), float(grey.std() / 255.0),
    ], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--tag", default="photometa")
    args = parser.parse_args()

    df = pd.read_csv(DATA_DIR / f"{args.split}.csv")
    image_dir = DATA_DIR / args.split
    feats = np.stack([
        describe(image_dir / str(i))
        for i in tqdm(df["imageid"], desc=args.split, unit="img", leave=False)
    ])

    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    out = FEATURE_DIR / f"{args.tag}__{args.split}.npz"
    np.savez_compressed(out, features=feats, imageid=df["imageid"].to_numpy())
    (FEATURE_DIR / f"{args.tag}__{args.split}.json").write_text(
        json.dumps({"names": NAMES, "shape": list(feats.shape), "split": args.split}, indent=2))
    print(f"wrote {out}  shape {feats.shape}")

    if args.split == "train" and "price" in df.columns:
        # A quick univariate read on whether any of this correlates at all.
        price = df["price"].to_numpy(dtype=np.float64)
        corr = [(n, float(np.corrcoef(feats[:, i], np.log(price))[0, 1]))
                for i, n in enumerate(NAMES)]
        corr.sort(key=lambda kv: -abs(kv[1]))
        print("\nSpearman-free check: Pearson corr with log price, strongest first")
        for n, c in corr:
            print(f"  {n:<18} {c:+.3f}")


if __name__ == "__main__":
    main()
