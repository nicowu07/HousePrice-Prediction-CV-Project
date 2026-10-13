"""Survey whatever the competition archive actually contains.

Run once right after download. It makes no assumptions about the directory
layout: it walks data/raw/, reports every CSV and every image folder it finds,
and summarises the price distribution plus the trivial constant-prediction
baseline (predicting the training mean), which is the reference point the
Kaggle marking scheme calls "a simple baseline".

Usage
-----
    python src/inspect_data.py
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def report_csvs(root: Path) -> list[Path]:
    csvs = sorted(root.rglob("*.csv"))
    print(f"\n{'=' * 70}\nCSV FILES ({len(csvs)})\n{'=' * 70}")
    for path in csvs:
        df = pd.read_csv(path)
        print(f"\n{path.relative_to(root)}  ->  {df.shape[0]} rows x {df.shape[1]} cols")
        print(f"  columns: {list(df.columns)}")
        print(df.head(3).to_string(index=False, max_colwidth=28))
    return csvs


def report_images(root: Path) -> dict[Path, list[Path]]:
    """Group every image under root by its parent directory."""
    by_dir: dict[Path, list[Path]] = {}
    for path in root.rglob("*"):
        if path.suffix.lower() in IMAGE_SUFFIXES:
            by_dir.setdefault(path.parent, []).append(path)

    print(f"\n{'=' * 70}\nIMAGE FOLDERS ({len(by_dir)})\n{'=' * 70}")
    for folder, files in sorted(by_dir.items()):
        # Sampling keeps this fast on folders with tens of thousands of files.
        sample = files[:: max(1, len(files) // 200)][:200]
        sizes, modes, bytes_ = [], Counter(), []
        for path in sample:
            try:
                with Image.open(path) as img:
                    sizes.append(img.size)
                    modes[img.mode] += 1
                bytes_.append(path.stat().st_size)
            except Exception as exc:  # corrupt files matter for the integrity check
                print(f"  [warn] unreadable: {path.name} ({exc})")

        widths = np.array([s[0] for s in sizes])
        heights = np.array([s[1] for s in sizes])
        print(f"\n{folder.relative_to(root)}  ->  {len(files)} images")
        print(f"  sampled {len(sizes)}: width {widths.min()}-{widths.max()} "
              f"(median {int(np.median(widths))}), "
              f"height {heights.min()}-{heights.max()} (median {int(np.median(heights))})")
        print(f"  distinct sizes in sample: {len(set(sizes))}  |  modes: {dict(modes)}")
        print(f"  median file size: {int(np.median(bytes_)) / 1024:.0f} KB")
        print(f"  example names: {[p.name for p in files[:5]]}")
    return by_dir


def report_prices(csvs: list[Path], root: Path) -> None:
    """Summarise any CSV that carries a price column, and give the MSE floor."""
    print(f"\n{'=' * 70}\nPRICE DISTRIBUTION\n{'=' * 70}")
    for path in csvs:
        df = pd.read_csv(path)
        price_cols = [c for c in df.columns if "price" in c.lower()]
        if not price_cols:
            continue
        col = price_cols[0]
        prices = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy()
        if prices.size == 0 or np.unique(prices).size <= 1:
            continue  # sample-solution.csv is a constant placeholder

        qs = np.percentile(prices, [0, 1, 5, 25, 50, 75, 95, 99, 100])
        print(f"\n{path.relative_to(root)}  column '{col}'  n={prices.size}")
        print(f"  mean {prices.mean():,.1f}  std {prices.std():,.1f}  "
              f"skew {pd.Series(prices).skew():.2f}")
        print("  percentiles  " + "  ".join(
            f"p{p}={v:,.0f}" for p, v in zip([0, 1, 5, 25, 50, 75, 95, 99, 100], qs)))

        # The metric is MSE on raw price, so predicting the mean gives MSE = variance.
        # Every model we build has to beat this number to be worth anything.
        var = prices.var()
        print(f"\n  [baseline] predict-the-mean MSE = {var:,.0f}  (RMSE {np.sqrt(var):,.1f})")

        # How much of that error sits in the expensive tail? This is the premise
        # of our whole method, so we re-derive it on the real data rather than
        # trusting the number from the early exploration.
        for thresh in (1000, 1500):
            if (prices >= thresh).any():
                mask = prices >= thresh
                sq = (prices - prices.mean()) ** 2
                print(f"  [tail] price >= {thresh}: {mask.mean() * 100:.1f}% of rows, "
                      f"{sq[mask].sum() / sq.sum() * 100:.1f}% of total squared error")

        log_prices = np.log(prices)
        print(f"  [log] mean {log_prices.mean():.3f}  std {log_prices.std():.3f}  "
              f"skew {pd.Series(log_prices).skew():.2f}  "
              f"(closer to 0 means log is the better training scale)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=RAW_DIR)
    args = parser.parse_args()

    if not args.root.exists() or not any(args.root.iterdir()):
        raise SystemExit(f"{args.root} is empty - run src/download_data.py first")

    print(f"Inspecting {args.root}")
    csvs = report_csvs(args.root)
    report_images(args.root)
    report_prices(csvs, args.root)


if __name__ == "__main__":
    main()
