"""Build the fixed cross-validation folds, with near-duplicate leakage control.

Two decisions are frozen here and never revisited, because changing a split
mid-project silently invalidates every earlier comparison:

1.  **Near-duplicate grouping.** Real-estate listings are re-posted, and the
    same property can appear more than once with a slightly different crop,
    watermark or JPEG re-encode. If such a pair straddles a fold boundary the
    model can memorise rather than generalise, and validation error becomes
    optimistic. We detect duplicates with a perceptual hash (pHash), union
    them into groups, and keep every group entirely within one fold.

2.  **Price-stratified grouping.** Prices are heavily right-skewed, so a plain
    random split gives folds with noticeably different tail composition. Since
    the metric is raw-price MSE and the tail dominates it, that alone would
    swamp the differences between the methods we want to compare. We stratify
    on price decile so every fold sees a comparable tail.

Output: data/splits/folds.csv with columns imageid, price, group, fold.

Usage
-----
    python src/make_splits.py --train-csv data/raw/train.csv \
                              --image-dir data/resized/train
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "data" / "splits" / "folds.csv"


def phash(path: Path, hash_size: int = 8) -> np.uint64:
    """64-bit perceptual hash: DCT of a 32x32 greyscale thumbnail, low
    frequencies thresholded at their median. Robust to re-encoding, rescaling
    and mild colour shifts, which is exactly the duplicate family we expect."""
    import imagehash

    with Image.open(path) as img:
        h = imagehash.phash(img, hash_size=hash_size)
    bits = h.hash.flatten()
    return np.uint64(int("".join("1" if b else "0" for b in bits), 2))


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, a: int) -> int:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def group_duplicates(hashes: np.ndarray, max_distance: int, chunk: int = 512) -> np.ndarray:
    """Union images whose pHash Hamming distance is <= max_distance.

    Compared chunk-wise so the 64-bit pairwise XOR never materialises as a
    single n x n array."""
    n = len(hashes)
    uf = UnionFind(n)
    pairs = 0
    for start in tqdm(range(0, n, chunk), desc="duplicate scan", unit="chunk"):
        block = hashes[start : start + chunk, None]
        # Only compare against later images: the matrix is symmetric.
        dist = np.bitwise_count(np.bitwise_xor(block, hashes[None, start:]))
        rows, cols = np.nonzero(dist <= max_distance)
        cols = cols + start
        for i, j in zip(rows + start, cols):
            if i < j:
                uf.union(int(i), int(j))
                pairs += 1
    print(f"  {pairs} duplicate pairs at Hamming distance <= {max_distance}")
    return np.array([uf.find(i) for i in range(n)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--id-col", default="imageid")
    parser.add_argument("--price-col", default="price")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=86)
    parser.add_argument("--max-distance", type=int, default=6,
                        help="pHash Hamming distance counted as a duplicate")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    df = pd.read_csv(args.train_csv)
    if args.id_col not in df.columns or args.price_col not in df.columns:
        raise SystemExit(f"expected columns {args.id_col!r}/{args.price_col!r}, got {list(df.columns)}")
    df = df[[args.id_col, args.price_col]].rename(
        columns={args.id_col: "imageid", args.price_col: "price"}
    )
    print(f"{len(df)} training rows")

    # Resolve each id to a file, tolerating ids with or without an extension.
    paths = []
    missing = []
    for image_id in df["imageid"]:
        stem = Path(str(image_id)).stem
        path = args.image_dir / f"{stem}.jpg"
        if not path.exists():
            candidates = list(args.image_dir.glob(f"{stem}.*"))
            path = candidates[0] if candidates else None
        if path is None:
            missing.append(image_id)
        paths.append(path)
    if missing:
        raise SystemExit(f"{len(missing)} images not found in {args.image_dir}, e.g. {missing[:5]}")

    print("hashing images...")
    hashes = np.array(
        [phash(p) for p in tqdm(paths, unit="img")], dtype=np.uint64
    )

    df["group"] = group_duplicates(hashes, args.max_distance)
    n_groups = df["group"].nunique()
    print(f"  {len(df)} images -> {n_groups} groups "
          f"({len(df) - n_groups} images absorbed into a duplicate group)")

    # Stratify on price decile so the expensive tail is spread evenly.
    strata = pd.qcut(df["price"], q=10, labels=False, duplicates="drop")
    splitter = StratifiedGroupKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    df["fold"] = -1
    for fold, (_, val_idx) in enumerate(splitter.split(df, strata, groups=df["group"])):
        df.loc[df.index[val_idx], "fold"] = fold
    assert (df["fold"] >= 0).all(), "some rows were not assigned a fold"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"\nwrote {args.out}")
    summary = df.groupby("fold")["price"].agg(["count", "mean", "std", "max"])
    summary["pct>=1000"] = df.groupby("fold")["price"].apply(lambda s: (s >= 1000).mean() * 100)
    print(summary.to_string(float_format=lambda v: f"{v:,.1f}"))


if __name__ == "__main__":
    main()
