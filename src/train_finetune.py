"""End-to-end fine-tuning of a pretrained backbone.

The frozen-feature experiments answer a narrower question than the task asks:
how much price information happens to be present in a general-purpose
representation. The answer was R^2 ~ 0.50. But price is driven largely by
fine-grained quality judgements -- cladding and roofing material, build
quality, how well the landscaping is maintained, whether the paving is
cracked -- and a self-supervised image-level objective has no particular
reason to encode those well. Fine-tuning lets the gradient shape the
representation around them.

Design choices and why:

*   **Layer-wise learning-rate decay.** Early layers hold generic edge and
    texture filters that 8,000 images cannot improve on; late layers hold the
    semantics that must change. A uniform learning rate either destroys the
    former or fails to move the latter, and destroying it is the dominant
    overfitting mode when fine-tuning a large backbone on small data.

*   **Scalar head on log price.** The binned-distribution head was tested
    against this in E4 and lost decisively, including after a sweep over bin
    count and smoothing, so the main model does not use it.

*   **Restrained augmentation.** Horizontal flip and mild crop jitter are
    safe. Vertical flips and large rotations are not: listing photos have a
    strong canonical orientation and breaking it moves the input off the
    pretraining manifold. Colour jitter stays mild because exterior colour and
    lighting may themselves carry price signal -- aggressive jitter would
    delete a feature rather than regularise one.

*   **EMA weights.** Cheap variance reduction, which is exactly what a squared
    error metric rewards.

Usage
-----
    python src/train_finetune.py --folds 0 --epochs 20          # pilot
    python src/train_finetune.py --epochs 20 --write-submission # all folds
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image
from sklearn.isotonic import IsotonicRegression
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from common import (
    DATA_DIR,
    EXPERIMENT_DIR,
    band_errors,
    clip_prices,
    format_metrics,
    load_folds,
    metrics,
    reliability,
    write_submission,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

class HouseDataset(Dataset):
    """Images decoded once into RAM; augmentation applied per access.

    The whole training split is ~2.9 GB of uint8, so holding it in memory
    removes disk I/O entirely. Workers stay at 0 deliberately: on Windows they
    are spawned, and each would receive its own copy of that array.
    """

    def __init__(self, image_ids, prices, image_dir: Path, img_size: int, train: bool) -> None:
        self.images = [
            np.asarray(Image.open(image_dir / str(i)).convert("RGB"))
            for i in tqdm(image_ids, desc="decoding", unit="img", leave=False)
        ]
        self.prices = None if prices is None else np.asarray(prices, dtype=np.float32)
        self.img_size = img_size
        self.train = train
        # ImageNet statistics; DINOv2 and the ConvNeXt checkpoints both use them.
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

    def __len__(self) -> int:
        return len(self.images)

    def _augment(self, img: Image.Image) -> Image.Image:
        import random

        w, h = img.size
        # Crop jitter: keep most of the frame. E1 showed that preserving the
        # full width beats centre-cropping, so crops stay generous.
        scale = random.uniform(0.65, 1.0)
        ratio = random.uniform(0.85, 1.18)
        cw = min(w, int(round(w * np.sqrt(scale) * np.sqrt(ratio))))
        ch = min(h, int(round(h * np.sqrt(scale) / np.sqrt(ratio))))
        x0 = random.randint(0, w - cw)
        y0 = random.randint(0, h - ch)
        img = img.crop((x0, y0, x0 + cw, y0 + ch))

        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        if random.random() < 0.7:
            from PIL import ImageEnhance

            for enhancer, strength in (
                (ImageEnhance.Brightness, 0.15),
                (ImageEnhance.Contrast, 0.15),
                (ImageEnhance.Color, 0.12),
            ):
                img = enhancer(img).enhance(1.0 + random.uniform(-strength, strength))
        return img

    def __getitem__(self, idx: int):
        img = Image.fromarray(self.images[idx])
        if self.train:
            img = self._augment(img)
        # Squash to square: E1 found this beats centre-crop and letterbox,
        # because the full frame carries the driveway, planting and lot that
        # a centre crop discards.
        img = img.resize((self.img_size, self.img_size), Image.BICUBIC)

        arr = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        x = torch.from_numpy((arr - self.mean) / self.std)
        if self.prices is None:
            return x
        return x, torch.tensor(np.log(self.prices[idx]), dtype=torch.float32)


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------

def build_model(name: str, img_size: int) -> nn.Module:
    extra = {"img_size": img_size} if name.startswith("vit_") else {}
    return timm.create_model(name, pretrained=True, num_classes=1, **extra)


def param_groups(model: nn.Module, weight_decay: float, layer_decay: float):
    """Layer-wise learning-rate decay, via timm's layer assignment.

    timm records the decay as an `lr_scale` on each group, which timm's own
    schedulers honour but torch's do not -- a torch scheduler overwrites every
    group's `lr` from a scalar `max_lr` and the decay silently disappears. The
    per-group scales are therefore returned so the caller can hand OneCycleLR
    a list of maximum rates instead.
    """
    from timm.optim import param_groups_layer_decay

    groups = param_groups_layer_decay(
        model, weight_decay=weight_decay, layer_decay=layer_decay,
        no_weight_decay_list=model.no_weight_decay() if hasattr(model, "no_weight_decay") else (),
    )
    scales = [g.get("lr_scale", 1.0) for g in groups]
    return groups, scales


class EMA:
    """Exponential moving average of weights, evaluated instead of the raw model."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)

    def copy_to(self, model: nn.Module) -> dict:
        backup = {k: v.detach().clone() for k, v in model.state_dict().items() if k in self.shadow}
        model.load_state_dict({**model.state_dict(), **self.shadow}, strict=False)
        return backup


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, tta_flip: bool) -> np.ndarray:
    """Predictions in log space."""
    model.eval()
    out = []
    for batch in loader:
        x = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(DEVICE, non_blocking=True)
        with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
            p = model(x).float().squeeze(1)
            if tta_flip:
                p = (p + model(torch.flip(x, dims=[3])).float().squeeze(1)) / 2
        out.append(p.cpu())
    return torch.cat(out).numpy()


def train_fold(args, folds: pd.DataFrame, fold: int, test_ds: Dataset):
    va_mask = (folds["fold"] == fold).to_numpy()
    tr_rows, va_rows = folds[~va_mask], folds[va_mask]
    image_dir = DATA_DIR / "train"

    tr_ds = HouseDataset(tr_rows["imageid"], tr_rows["price"], image_dir, args.img_size, train=True)
    va_ds = HouseDataset(va_rows["imageid"], va_rows["price"], image_dir, args.img_size, train=False)
    tr_ld = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    va_ld = DataLoader(va_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=0)
    te_ld = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=0)

    torch.manual_seed(args.seed + fold)
    model = build_model(args.model, args.img_size).to(DEVICE)
    if args.grad_checkpoint and hasattr(model, "set_grad_checkpointing"):
        model.set_grad_checkpointing(True)
    groups, scales = param_groups(model, args.weight_decay, args.layer_decay)
    opt = torch.optim.AdamW(groups, lr=args.lr)
    steps = args.epochs * len(tr_ld)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[args.lr * s for s in scales], total_steps=steps, pct_start=0.1)
    ema = EMA(model, decay=args.ema_decay)
    loss_fn = nn.MSELoss()

    t0 = time.time()
    history: list[tuple[int, float, float]] = []
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for x, t in tqdm(tr_ld, desc=f"fold {fold} epoch {epoch + 1}/{args.epochs}", leave=False):
            x, t = x.to(DEVICE, non_blocking=True), t.to(DEVICE, non_blocking=True)
            if args.mixup > 0:
                # Mixing images and their log-prices linearly is coherent for
                # regression in a way it is not for classification: the target
                # of a blended image is the blended target.
                lam = float(np.random.beta(args.mixup, args.mixup))
                perm = torch.randperm(x.size(0), device=DEVICE)
                x = lam * x + (1 - lam) * x[perm]
                t = lam * t + (1 - lam) * t[perm]
            with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
                loss = loss_fn(model(x).squeeze(1), t)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            ema.update(model)
            running += loss.item()

        if (epoch + 1) % args.log_every == 0 or epoch == args.epochs - 1:
            ema.copy_to(model)
            val_log = predict(model, va_ld, tta_flip=False)
            val_mse = float(((np.exp(val_log) - va_rows["price"].to_numpy()) ** 2).mean())
            history.append((epoch + 1, running / len(tr_ld), val_mse))
            # flush: without it the whole run is a black box until it exits,
            # and an under-trained or diverging run looks identical to a
            # healthy one until the end.
            print(f"  fold {fold} epoch {epoch + 1:>3}  train_loss {running / len(tr_ld):.4f}  "
                  f"val_MSE(uncalibrated) {val_mse:,.0f}  [{time.time() - t0:.0f}s]", flush=True)

    ema.copy_to(model)
    val_log = predict(model, va_ld, tta_flip=args.tta_flip)
    test_log = predict(model, te_ld, tta_flip=args.tta_flip)
    return val_log, test_log, va_mask, history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="vit_base_patch14_reg4_dinov2.lvd142m")
    parser.add_argument("--img-size", type=int, default=280)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--layer-decay", type=float, default=0.75)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--tta-flip", action="store_true")
    parser.add_argument("--mixup", type=float, default=0.0,
                        help="Beta(a,a) mixup strength; 0 disables")
    parser.add_argument("--grad-checkpoint", action="store_true",
                        help="trade compute for activation memory on large backbones")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--folds", type=int, nargs="+", default=None, help="subset of folds (default: all)")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--write-submission", action="store_true",
                        help="write a Kaggle-format CSV locally (does not upload)")
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    folds = load_folds()
    y = folds["price"].to_numpy(dtype=np.float64)
    run_folds = sorted(folds["fold"].unique()) if args.folds is None else args.folds
    tag = args.tag or f"ft_{args.model.split('.')[0]}_{args.img_size}"
    print(f"{args.model} @ {args.img_size} | folds {run_folds} | device {DEVICE}")
    print(f"constant-prediction MSE = {y.var():,.0f}\n")

    test_df = pd.read_csv(DATA_DIR / "test.csv")
    test_ds = HouseDataset(test_df["imageid"], None, DATA_DIR / "test", args.img_size, train=False)

    oof_log = np.full(len(folds), np.nan)
    test_logs = []
    histories: dict[int, list] = {}
    for fold in run_folds:
        val_log, test_log, va_mask, hist = train_fold(args, folds, fold, test_ds)
        histories[fold] = hist
        oof_log[va_mask] = val_log
        test_logs.append(test_log)

    done = ~np.isnan(oof_log)
    # Same decoder as the frozen-feature line: isotonic, fitted on out-of-fold
    # predictions only, so the comparison between the two is like-for-like.
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(np.exp(oof_log[done]), y[done])
    oof_price = clip_prices(calibrator.predict(np.exp(oof_log[done])))

    m = metrics(y[done], oof_price)
    print("\n" + format_metrics(tag, m))
    print(f"evaluated on {done.sum()} of {len(folds)} rows "
          f"({100 * (1 - m['mse'] / y[done].var()):.1f}% of baseline error removed)")

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(EXPERIMENT_DIR / f"{tag}_oof.npz", oof_log=oof_log, mask=done,
             test_log=np.mean(test_logs, axis=0))
    (EXPERIMENT_DIR / f"{tag}_config.json").write_text(
        json.dumps({**vars(args), "mse": m["mse"], "r2": m["r2"],
                    "history": {str(k): v for k, v in histories.items()}},
                   indent=2, default=str))

    print("\nper-band error (binned by TRUE price):")
    print(band_errors(y[done], oof_price).to_string(index=False, float_format=lambda v: f"{v:,.1f}"))
    print("\nreliability (binned by PREDICTION):")
    print(reliability(y[done], oof_price).to_string(index=False, float_format=lambda v: f"{v:,.1f}"))

    if args.write_submission:
        test_price = calibrator.predict(np.exp(np.mean(test_logs, axis=0)))
        write_submission(test_df["imageid"].to_numpy(), test_price, tag)


if __name__ == "__main__":
    main()
