"""Second-round report figures.

Kept separate from make_figures.py so the two rounds' figures can be
regenerated independently. Style, palette and column widths are imported from
make_figures so both sets stay visually identical.

The three panels here each carry a finding that a table states less forcefully:
what actually moves the representation, why the optimal input resolution is a
property of the data, and why ensemble weight ignores member accuracy.

Usage
-----
    python src/figures_part2.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from make_figures import (
    AQUA, AXIS, BLUE, COL1, FIG_DIR, INK, INK2, MUTED, ORANGE, SEQ, SURFACE,
    setup, tidy,
)


def fig_factors() -> None:
    """Pretraining objective, architecture and capacity, each isolated.

    Every pair differs in exactly one factor, which is what makes the
    magnitudes comparable. Capacity is the lever most often reached for first,
    and here it is worth almost nothing.
    """
    pairs = [
        ("Pretraining objective\nConvNeXt-B: IN-22k sup. to DINOv3 SSL", 98752, 87744),
        ("Architecture\nDINOv3: ConvNeXt-L to ViT-L", 86454, 76802),
        ("Capacity\nDINOv3 ConvNeXt: B to L", 87744, 86454),
    ]
    labels = [p[0] for p in pairs]
    deltas = [100 * (p[2] / p[1] - 1) for p in pairs]

    fig, ax = plt.subplots(figsize=(COL1, 2.0))
    tidy(ax, grid_axis="x")
    bars = ax.barh(range(len(pairs)), deltas, color=SEQ[2], height=0.55, zorder=3)
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels(labels, fontsize=6.8, linespacing=1.35)
    ax.set_xlabel("change in out-of-fold MSE")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_xlim(min(deltas) * 1.3, 0)
    ax.axvline(0, color=AXIS, lw=0.8, zorder=2)
    for b, d in zip(bars, deltas):
        # These bars grow leftwards from zero, so their data end is at x = d.
        # A long bar can hold the value inside, offset to the right of that
        # tip; a short one cannot, and takes it outside to the left.
        fits_inside = abs(d) > 4.0
        ax.annotate(f"{d:.1f}%", (d, b.get_y() + b.get_height() / 2),
                    textcoords="offset points",
                    xytext=(5, 0) if fits_inside else (-5, 0),
                    ha="left" if fits_inside else "right", va="center",
                    fontsize=7, color=SURFACE if fits_inside else INK2)
    fig.savefig(FIG_DIR / "fig5_factors.pdf")
    plt.close(fig)


def fig_resolution() -> None:
    """Two families whose pretraining resolutions differ by an octave, both
    optimising at the same input size.

    The rings mark each model's own native resolution; neither is its optimum.
    The source photographs are 400x300, so past roughly 336 there is nothing
    left to resolve and further upsampling only interpolates.
    """
    series = [
        ("DINOv2 ViT-L/14 (native 518)", [224, 336, 518], [78078, 76807, 77231], BLUE, 518),
        ("DINOv3 ViT-L/16 (native 256)", [224, 256, 336], [76802, 76112, 74134], ORANGE, 256),
    ]
    fig, ax = plt.subplots(figsize=(COL1, 2.5))
    tidy(ax, grid_axis="both")
    for name, xs, ys, colour, native in series:
        ax.plot(xs, ys, "-o", color=colour, ms=4.5, mec=SURFACE, mew=0.9,
                zorder=4, label=name)
        ax.plot([native], [ys[xs.index(native)]], "o", ms=11, mfc="none",
                mec=colour, mew=1.2, zorder=5)

    ax.axvline(336, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=2)
    # Vertical, hugging the rule: the plot area is too crowded for a
    # horizontal caption anywhere near the line.
    ax.annotate("source photo is 400x300", (336, 0.42), xycoords=("data", "axes fraction"),
                textcoords="offset points", xytext=(-4, 0), rotation=90,
                ha="right", va="center", fontsize=6.4, color=MUTED)
    ax.set_xlabel("input resolution (px)")
    ax.set_ylabel("out-of-fold MSE")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    # Legend outside the axes: every in-plot position collided with a series.
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=1, fontsize=6.6, handletextpad=0.5)
    fig.savefig(FIG_DIR / "fig6_resolution.pdf")
    plt.close(fig)


def fig_ensemble() -> None:
    """Blend weight against each member's own error.

    If ensembling rewarded accuracy, the points would fall to the right. They
    do not. The two heaviest members are among the weakest, and they are the
    only member from a different pretraining family and the only one with a
    different input geometry. Weight follows decorrelation, not accuracy.
    """
    members = [
        ("DINOv2 ViT-L @518", 77231, 0.193, True),
        ("DINOv3 ViT-H+ @256", 71344, 0.169, False),
        ("H+ @224+@336", 70542, 0.163, False),
        ("H+@336 + L@336 + meta", 69688, 0.141, False),
        ("DINOv3 ViT-L 2x2 tiles", 76800, 0.129, True),
        ("DINOv3 ViT-L @336", 74134, 0.076, False),
        ("H+@336 + L@336 + tiles + meta", 69832, 0.068, False),
        ("DINOv3 ViT-H+ @336", 70818, 0.037, False),
        ("6-way concatenation", 69175, 0.033, False),
    ]
    fig, ax = plt.subplots(figsize=(COL1, 2.3))
    tidy(ax, grid_axis="both")
    for _, mse, w, distinct in members:
        ax.plot([mse], [w], "o", ms=6, color=ORANGE if distinct else BLUE,
                mec=SURFACE, mew=1.0, zorder=4)

    # Label only the three points that carry the argument; the left-hand
    # cluster is too dense to label without collisions, and the legend plus
    # the trend already say what those points are.
    for name, mse, w, dx, dy, ha in [
        ("DINOv2 ViT-L @518", 77231, 0.193, 0, -14, "center"),
        ("DINOv3 ViT-L 2x2 tiles", 76800, 0.129, -8, 0, "right"),
        ("6-way concat\n(best single member)", 69175, 0.033, 2, -15, "center"),
    ]:
        ax.annotate(name, (mse, w), textcoords="offset points",
                    xytext=(dx, dy), ha=ha, va="center", fontsize=6.2,
                    color=INK2, linespacing=1.2)

    ax.set_xlabel("member's own out-of-fold MSE")
    ax.set_ylabel("blend weight")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    ax.set_ylim(0, 0.235)
    ax.set_xlim(68300, 78600)
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", color=ORANGE, ms=5,
               label="only member of its kind"),
        Line2D([], [], marker="o", ls="", color=BLUE, ms=5,
               label="DINOv3 global variant"),
    ], frameon=False, loc="upper left", fontsize=6.6)
    fig.savefig(FIG_DIR / "fig7_ensemble_weights.pdf")
    plt.close(fig)


def main() -> None:
    setup()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_factors()
    fig_resolution()
    fig_ensemble()
    for p in sorted(FIG_DIR.glob("fig[567]*.pdf")):
        print(f"wrote {p.relative_to(FIG_DIR.parents[1])}  ({p.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
