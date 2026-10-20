"""Generate the report figures from saved experiment outputs.

Vector PDF at IEEE two-column widths (3.4 in single column, 7.0 in full), so
they drop straight into LaTeX without rescaling and the text stays selectable.

Colour follows the project's data-viz reference palette. Only the first three
categorical slots are used, which is the documented all-pairs colour-blind-safe
subset; ordered quantities use the single-hue blue sequential ramp rather than
distinct hues, because the thing being encoded is magnitude, not identity.
Grid and axes are recessive hairlines; values are labelled directly where it
replaces a lookup, never on every mark.

Usage
-----
    python src/make_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import EXPERIMENT_DIR, PRICE_BANDS, band_errors, load_folds, reliability

FIG_DIR = Path(__file__).resolve().parents[1] / "report" / "figures"

# --- palette (reference instance; slots 1-3 only) -------------------------
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
# single-hue sequential ramp, light -> dark, for ordered magnitude
SEQ = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

COL1, COL2 = 3.4, 7.0  # IEEE column widths, inches


def setup() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.labelcolor": INK2,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.6,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.6,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def tidy(ax, grid_axis="y") -> None:
    """Recessive chrome: no box, hairline grid behind the marks."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.grid(True, axis=grid_axis, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def load_best_oof() -> tuple[np.ndarray, np.ndarray]:
    """True prices and the best model's out-of-fold predictions."""
    folds = load_folds()
    y = folds["price"].to_numpy(dtype=np.float64)
    data = np.load(EXPERIMENT_DIR / "best_concat_oof.npz", allow_pickle=True)
    key = next(k for k in data.files if data[k].shape == y.shape)
    return y, data[key]


# --------------------------------------------------------------------------
# Figure 1 - the calibration result
# --------------------------------------------------------------------------

def fig_calibration(y, pred) -> None:
    """Two panels that answer the same question and disagree.

    Conditioning on the true price (left) makes the model look badly biased on
    expensive houses. Conditioning on the prediction (right) shows it is
    calibrated. The pair is the argument: the apparent tail bias is regression
    toward the mean, not a defect to be corrected.
    """
    bands = band_errors(y, pred)
    rel = reliability(y, pred)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COL2, 2.5))

    # left: mean signed error within each true-price band
    tidy(ax1)
    bars = ax1.bar(range(len(bands)), bands["bias"], color=SEQ[2], width=0.62, zorder=3)
    ax1.axhline(0, color=AXIS, lw=0.8, zorder=2)
    ax1.set_xticks(range(len(bands)))
    ax1.set_xticklabels(bands["band"], rotation=20, ha="right")
    ax1.set_xlabel("true price band  ($1000)")
    ax1.set_ylabel("mean signed error  ($1000)")
    ax1.set_title("(a) binned by true price", loc="left", color=INK)
    span = bands["bias"].max() - bands["bias"].min()
    for b, v in zip(bars, bands["bias"]):
        # A label below a deep negative bar would land on the rotated tick
        # labels, so tall bars carry their value inside instead.
        inside = abs(v) > 0.18 * span
        if inside:
            off, colour, va = (0, -9 if v > 0 else 9), SURFACE, "center"
        else:
            off, colour, va = (0, 3 if v > 0 else -10), INK2, "center"
        ax1.annotate(f"{v:+.0f}", (b.get_x() + b.get_width() / 2, v),
                     textcoords="offset points", xytext=off, ha="center",
                     va=va, fontsize=7, color=colour)

    # right: mean outcome against mean prediction, per prediction decile
    tidy(ax2, grid_axis="both")
    lo = min(rel["mean_pred"].min(), rel["mean_true"].min()) - 60
    hi = max(rel["mean_pred"].max(), rel["mean_true"].max()) + 60
    ax2.plot([lo, hi], [lo, hi], color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=2)
    ax2.plot(rel["mean_pred"], rel["mean_true"], "o", color=BLUE, ms=5,
             mec=SURFACE, mew=1.0, zorder=4)
    ax2.set_xlim(lo, hi)
    ax2.set_ylim(lo, hi)
    ax2.set_aspect("equal")
    ax2.set_xlabel("mean predicted price  ($1000)")
    ax2.set_ylabel("mean actual price  ($1000)")
    ax2.set_title("(b) binned by prediction", loc="left", color=INK)
    ax2.annotate("perfect calibration", (hi - 40, hi - 40), textcoords="offset points",
                 xytext=(-6, -14), ha="right", fontsize=7, color=MUTED)

    fig.savefig(FIG_DIR / "fig1_calibration.pdf")
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 2 - where the squared error lives
# --------------------------------------------------------------------------

def fig_bands(y, pred) -> None:
    """MSE per price band, with each band's share of total squared error as a
    direct label. Two measures of different scale on one axis would need a
    second y-scale; labelling the share instead keeps a single axis."""
    bands = band_errors(y, pred)
    fig, ax = plt.subplots(figsize=(COL1, 2.3))
    tidy(ax)
    bars = ax.bar(range(len(bands)), bands["mse"], color=SEQ, width=0.62, zorder=3)
    ax.set_xticks(range(len(bands)))
    ax.set_xticklabels([b.replace("-", "–") for b in bands["band"]],
                       rotation=20, ha="right")
    ax.set_xlabel("true price band  ($1000)")
    ax.set_ylabel("MSE")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    for b, (_, r) in zip(bars, bands.iterrows()):
        ax.annotate(f"{r['pct_total_se']:.0f}% of\ntotal SE",
                    (b.get_x() + b.get_width() / 2, b.get_height()),
                    textcoords="offset points", xytext=(0, 3), ha="center",
                    fontsize=6.8, color=INK2, linespacing=1.15)
    ax.set_ylim(0, bands["mse"].max() * 1.28)
    fig.savefig(FIG_DIR / "fig2_price_bands.pdf")
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 3 - the weighting sweep
# --------------------------------------------------------------------------

def fig_objective(y) -> None:
    """MSE and effective sample size against the weighting exponent.

    These are different quantities on different scales, so they get stacked
    panels sharing one x axis rather than a second y-scale. The point of the
    pairing is that the MSE curve turns upward exactly where n_eff starts to
    collapse.
    """
    path = EXPERIMENT_DIR / "e5_objective.csv"
    if not path.exists():
        print("skip fig3: e5_objective.csv missing")
        return
    df = pd.read_csv(path)
    sweep = df[df["condition"].str.startswith("alpha")].copy()
    sweep["alpha"] = sweep["condition"].str[5:].astype(float)
    base = df.loc[df["condition"] == "A"]
    if len(base):
        sweep = pd.concat([
            pd.DataFrame({"alpha": [0.0], "mse": [float(base["mse"].iloc[0])]}), sweep
        ], ignore_index=True)
    sweep = sweep.sort_values("alpha")

    alphas = sweep["alpha"].to_numpy()
    n_eff = np.array([(w := y ** a).sum() ** 2 / (w ** 2).sum() for a in alphas])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(COL1, 3.3), sharex=True,
                                   gridspec_kw={"height_ratios": [1.5, 1]})
    tidy(ax1)
    ax1.plot(alphas, sweep["mse"], "-o", color=BLUE, ms=4,
             mec=SURFACE, mew=0.9, zorder=4)
    ax1.set_ylabel("out-of-fold MSE")
    ax1.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")

    best = sweep.loc[sweep["mse"].idxmin()]
    ax1.annotate(f"best  $\\alpha$={best['alpha']:.1f}",
                 (best["alpha"], best["mse"]), textcoords="offset points",
                 xytext=(6, 8), fontsize=7, color=INK2)
    ax1.annotate("metric-aligned\n$\\alpha$=2",
                 (2.0, float(sweep.loc[sweep["alpha"] == 2.0, "mse"].iloc[0])),
                 textcoords="offset points", xytext=(-8, 6), ha="right",
                 fontsize=7, color=MUTED, linespacing=1.15)

    tidy(ax2)
    ax2.plot(alphas, n_eff, "-o", color=ORANGE, ms=4, mec=SURFACE, mew=0.9, zorder=4)
    ax2.set_ylabel("effective\nsample size")
    ax2.set_xlabel("weighting exponent  $\\alpha$   ($w \\propto p^{\\alpha}$)")
    ax2.set_ylim(0, len(y) * 1.1)
    ax2.annotate(f"{n_eff[0]:,.0f}", (alphas[0], n_eff[0]), textcoords="offset points",
                 xytext=(4, -10), fontsize=7, color=INK2)
    ax2.annotate(f"{n_eff[-1]:,.0f}", (alphas[-1], n_eff[-1]), textcoords="offset points",
                 xytext=(-2, 8), ha="right", fontsize=7, color=INK2)

    fig.savefig(FIG_DIR / "fig3_weighting.pdf")
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 4 - representation comparison
# --------------------------------------------------------------------------

LABELS = {
    "vit_base_patch14_reg4_dinov2_224_squash": "DINOv2 ViT-B/14 @224",
    "vit_base_patch14_reg4_dinov2_280_squash": "DINOv2 ViT-B/14 @280",
    "vit_base_patch14_reg4_dinov2_336_squash": "DINOv2 ViT-B/14 @336",
    "vit_large_patch14_reg4_dinov2_224_squash": "DINOv2 ViT-L/14 @224",
    "vit_large_patch14_reg4_dinov2_336_squash": "DINOv2 ViT-L/14 @336",
    "convnext_base_288_squash": "ConvNeXt-B @288 (IN-22k)",
}


def fig_features() -> None:
    frames = [pd.read_csv(EXPERIMENT_DIR / f) for f in ("e3_features.csv", "e3b_concat.csv")
              if (EXPERIMENT_DIR / f).exists()]
    if not frames:
        print("skip fig4: feature sweep CSVs missing")
        return
    df = pd.concat(frames, ignore_index=True)
    df = df[~df["tag"].str.contains(r"\+|flip", regex=True)].drop_duplicates("tag")
    df["label"] = df["tag"].map(LABELS).fillna(df["tag"])
    df = df.sort_values("mse", ascending=False)

    # one colour: a single series encodes magnitude by bar length, not by hue
    fig, ax = plt.subplots(figsize=(COL1, 2.1))
    tidy(ax, grid_axis="x")
    bars = ax.barh(range(len(df)), df["mse"], color=SEQ[2], height=0.62, zorder=3)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["label"])
    ax.set_xlabel("out-of-fold MSE")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    ax.set_xlim(0, df["mse"].max() * 1.18)
    for b, v in zip(bars, df["mse"]):
        ax.annotate(f"{v:,.0f}", (b.get_width(), b.get_y() + b.get_height() / 2),
                    textcoords="offset points", xytext=(4, 0), va="center",
                    fontsize=7, color=INK2)
    fig.savefig(FIG_DIR / "fig4_representations.pdf")
    plt.close(fig)


def main() -> None:
    setup()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    y, pred = load_best_oof()
    print(f"best OOF predictions loaded: n={len(y)}  MSE={((pred - y) ** 2).mean():,.0f}")
    fig_calibration(y, pred)
    fig_bands(y, pred)
    fig_objective(y)
    fig_features()
    for p in sorted(FIG_DIR.glob("*.pdf")):
        print(f"wrote {p.relative_to(FIG_DIR.parents[1])}  ({p.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
