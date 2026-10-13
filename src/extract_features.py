"""Cache frozen-backbone features for the whole dataset.

The images are small (400x300 at most) and there are only 11,000 of them, so a
full pass costs seconds on the GPU. Caching the features turns every downstream
experiment -- target-scale comparison, calibration, regressor choice, meta
feature ablation -- into a few seconds of CPU work on a matrix, instead of a
GPU training run. That is what makes a wide experimental sweep affordable.

Two representation choices are baked in and worth stating:

*   **CLS + mean patch token, concatenated.** The CLS token summarises the
    image globally, which is the right granularity for holistic "curb appeal".
    The mean patch token averages local evidence -- cladding material, roof
    type, planting, paving -- which the CLS token compresses away. They are
    complementary, and concatenating costs nothing.

*   **Aspect-ratio handling is a flag, not an assumption.** `center` follows
    the usual short-side-resize-then-centre-crop recipe, which matches how the
    backbone was pretrained but discards ~31% of the width of a 4:3 listing
    photo -- potentially the garage, driveway or planting that carry the
    signal. `squash` resizes the full frame to a square, keeping all content at
    the cost of a 1.44x distortion that sits inside the aspect jitter DINOv2
    saw during pretraining. Which one wins is an empirical question; extract
    both and let the validation score decide.

Usage
-----
    python src/extract_features.py --split train --crop-mode squash
    python src/extract_features.py --split test  --crop-mode squash
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "house_dataset"
FEATURE_DIR = PROJECT_ROOT / "features"


class ImageList(Dataset):
    """Decodes every image once into RAM.

    8,000 training photos at 400x300x3 is about 2.9 GB uint8, which fits
    comfortably and removes disk I/O from the loop entirely. On Windows, where
    DataLoader workers are spawned rather than forked, this is faster than any
    worker configuration reading from disk.
    """

    def __init__(self, paths: list[Path], img_size: int, crop_mode: str,
                 mean: tuple[float, ...], std: tuple[float, ...]) -> None:
        self.img_size = img_size
        self.crop_mode = crop_mode
        self.mean = np.array(mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(std, dtype=np.float32).reshape(3, 1, 1)
        self.images = [
            np.asarray(Image.open(p).convert("RGB"))
            for p in tqdm(paths, desc="decoding", unit="img", leave=False)
        ]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.fromarray(self.images[idx])
        size = self.img_size

        if self.crop_mode == "squash":
            img = img.resize((size, size), Image.BICUBIC)
        elif self.crop_mode == "center":
            w, h = img.size
            scale = size / min(w, h)
            img = img.resize((max(size, round(w * scale)), max(size, round(h * scale))), Image.BICUBIC)
            w, h = img.size
            left, top = (w - size) // 2, (h - size) // 2
            img = img.crop((left, top, left + size, top + size))
        elif self.crop_mode == "pad":
            # Preserve aspect ratio and content by letterboxing onto grey.
            w, h = img.size
            scale = size / max(w, h)
            img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BICUBIC)
            canvas = Image.new("RGB", (size, size), (124, 116, 104))
            canvas.paste(img, ((size - img.size[0]) // 2, (size - img.size[1]) // 2))
            img = canvas
        else:
            raise ValueError(f"unknown crop mode {self.crop_mode!r}")

        arr = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return torch.from_numpy((arr - self.mean) / self.std)


@torch.no_grad()
def pool_features(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Global descriptor for either a ViT or a CNN backbone."""
    feats = model.forward_features(x)
    if feats.ndim == 4:  # CNN: B, C, H, W -> average + max pooled, concatenated
        return torch.cat([feats.mean(dim=(2, 3)), feats.amax(dim=(2, 3))], dim=1)
    n_prefix = getattr(model, "num_prefix_tokens", 1)  # DINOv2-reg has CLS + 4 registers
    return torch.cat([feats[:, 0], feats[:, n_prefix:].mean(dim=1)], dim=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--model", default="vit_base_patch14_reg4_dinov2.lvd142m")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--crop-mode", choices=["squash", "center", "pad"], default="squash")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--flip-tta", action="store_true",
                        help="average features of the image and its mirror")
    parser.add_argument("--tag", default=None, help="output filename stem")
    args = parser.parse_args()

    csv = DATA_DIR / f"{args.split}.csv"
    image_dir = DATA_DIR / args.split
    df = pd.read_csv(csv)
    paths = [image_dir / str(i) for i in df["imageid"]]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"{len(missing)} images missing, e.g. {missing[:3]}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    extra = {"img_size": args.img_size} if "vit_" in args.model else {}
    model = timm.create_model(args.model, pretrained=True, num_classes=0, **extra)
    model.eval().to(device)
    cfg = timm.data.resolve_data_config({}, model=model)
    print(f"{args.model} | img_size {args.img_size} | crop {args.crop_mode} | device {device}")

    dataset = ImageList(paths, args.img_size, args.crop_mode, cfg["mean"], cfg["std"])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    chunks = []
    with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
        for batch in tqdm(loader, desc="extracting", unit="batch", leave=False):
            batch = batch.to(device, non_blocking=True)
            out = pool_features(model, batch)
            if args.flip_tta:
                out = (out + pool_features(model, torch.flip(batch, dims=[3]))) / 2
            chunks.append(out.float().cpu())

    features = torch.cat(chunks).numpy().astype(np.float32)

    tag = args.tag or f"{args.model.split('.')[0]}_{args.img_size}_{args.crop_mode}" + ("_flip" if args.flip_tta else "")
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FEATURE_DIR / f"{tag}__{args.split}.npz"
    np.savez_compressed(out_path, features=features, imageid=df["imageid"].to_numpy())
    meta = {
        "model": args.model, "img_size": args.img_size, "crop_mode": args.crop_mode,
        "flip_tta": args.flip_tta, "split": args.split, "shape": list(features.shape),
    }
    (FEATURE_DIR / f"{tag}__{args.split}.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out_path}  shape {features.shape}")


if __name__ == "__main__":
    main()
