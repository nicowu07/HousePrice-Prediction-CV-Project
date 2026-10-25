"""Build report figures from the final ensemble OOF export and original photos.

Requires numpy, matplotlib and Pillow. Run from any working directory.
This script does not edit reportdraft.md or any source data / photographs.
"""

from pathlib import Path
import csv
import hashlib
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "report/images"
SOURCE = ROOT / "teammate/HousePrice-Prediction-CV-Project/outputs/experiments/oof_predictions_ensemble.csv"
CASE_IDS = ["4244.jpg", "2138.jpg", "5350.jpg", "2409.jpg"]
BLUE = "#285F8F"
ORANGE = "#C97732"
RED = "#B44940"
GREY = "#69717B"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_figure(fig, stem, svg=False):
    fig.savefig(OUT / f"{stem}.png", dpi=300, facecolor="white")
    fig.savefig(OUT / f"{stem}.pdf", facecolor="white",
                metadata={"Title": stem, "Creator": "COMP90086 report figure builder"})
    if svg:
        fig.savefig(OUT / f"{stem}.svg", facecolor="white")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with SOURCE.open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 8000
    assert len({r["imageid"] for r in rows}) == len(rows)
    assert {r["model"] for r in rows} == {"nine-member NNLS ensemble (frozen features)"}
    true = np.array([float(r["price"]) for r in rows])
    pred = np.array([float(r["pred"]) for r in rows])
    error = pred - true
    squared = error ** 2
    edges = [195, 400, 700, 1000, 1500, 2000]
    band_labels = ["195-<400", "400-<700", "700-<1,000", "1,000-<1,500", "1,500-2,000"]
    bands = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        selected = (true >= lo) & ((true < hi) if i < 4 else (true <= hi))
        bands.append({
            "band": band_labels[i], "lower": lo, "upper": hi,
            "upper_inclusive": i == 4, "n": int(selected.sum()),
            "sample_share_percent": float(100 * selected.mean()),
            "squared_error_share_percent": float(100 * squared[selected].sum() / squared.sum()),
            "mse": float(squared[selected].mean()),
            "mae": float(np.abs(error[selected]).mean()),
            "bias_prediction_minus_true": float(error[selected].mean()),
        })
    assert sum(b["n"] for b in bands) == 8000

    # Stable sorting makes equal-prediction tie handling reproducible.
    bins = []
    for i, indices in enumerate(np.array_split(np.argsort(pred, kind="stable"), 10), 1):
        bins.append({"bin": i, "n": len(indices),
                     "mean_prediction": float(pred[indices].mean()),
                     "mean_true_price": float(true[indices].mean())})

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.titlesize": 10, "axes.titleweight": "bold",
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#9299A0", "axes.labelcolor": "#222222",
        "text.color": "#222222", "grid.color": "#E4E7EA",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.05),
                             gridspec_kw={"width_ratios": [1.15, 1.2, 1]})
    fig.subplots_adjust(left=0.055, right=0.985, top=0.85, bottom=0.25, wspace=0.50)
    x = np.arange(5)
    ax = axes[0]
    ax.set_axisbelow(True)
    ax.grid(axis="y", linewidth=0.6)
    population = [b["sample_share_percent"] for b in bands]
    contribution = [b["squared_error_share_percent"] for b in bands]
    bars1 = ax.bar(x - 0.19, population, 0.36, label="Images", color=BLUE)
    bars2 = ax.bar(x + 0.19, contribution, 0.36, label="Squared error", color=ORANGE)
    ax.bar_label(bars1, labels=[f"{v:.1f}" for v in population], padding=3, fontsize=7.5)
    ax.bar_label(bars2, labels=[f"{v:.1f}" for v in contribution], padding=3, fontsize=7.5)
    ax.set_xticks(x, ["195-\n<400", "400-\n<700", "700-\n<1,000", "1,000-\n<1,500", "1,500-\n2,000"])
    ax.set_ylim(0, 50)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.set_xlabel("True price (US$1,000)")
    ax.set_ylabel("Share of total")
    ax.set_title("(a) Sample and error distribution", loc="left", pad=25)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(-0.02, 1.00), ncol=2, fontsize=8)

    ax = axes[1]
    biases = np.array([b["bias_prediction_minus_true"] for b in bands])
    ax.set_axisbelow(True)
    ax.grid(axis="x", linewidth=0.6)
    ax.barh(x, biases, height=0.60, color=[BLUE if v >= 0 else RED for v in biases])
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_yticks(x, [f"{b['band']}\n(n={b['n']:,})" for b in bands])
    ax.invert_yaxis()
    for i, value in enumerate(biases):
        ax.text(value + (12 if value >= 0 else -12), i, f"{value:+.1f}",
                ha="left" if value >= 0 else "right", va="center", fontsize=8)
    ax.set_xlim(-675, 210)
    ax.set_xticks([-600, -400, -200, 0, 200])
    ax.set_xlabel("Prediction - true price (US$1,000)")
    ax.set_title("(b) Mean signed error by true price", loc="left", pad=25)

    ax = axes[2]
    pmean = [b["mean_prediction"] for b in bins]
    ymean = [b["mean_true_price"] for b in bins]
    ax.plot([200, 1500], [200, 1500], linestyle="--", color=GREY, linewidth=1, label="Equal means")
    ax.plot(pmean, ymean, "o-", color=BLUE, markersize=4.5, linewidth=1, label="800 images per bin")
    ax.set_xlim(200, 1500)
    ax.set_ylim(200, 1500)
    ax.set_xticks([300, 600, 900, 1200, 1500])
    ax.set_yticks([300, 600, 900, 1200, 1500])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(linewidth=0.6)
    ax.set_xlabel("Mean prediction (US$1,000)")
    ax.set_ylabel("Mean true price (US$1,000)")
    ax.set_title("(c) Calibration by prediction bin", loc="left", pad=25)
    ax.legend(frameon=False, loc="upper left", fontsize=7.7)

    high = true >= 1000
    high_n = int(high.sum())
    high_share = float(100 * high.mean())
    high_sse_share = float(100 * squared[high].sum() / squared.sum())
    fig.text(0.5, 0.07,
             f"Nine-predictor blend | 8,000 OOF predictions | True price >= US$1 million: "
             f"{high_share:.1f}% of images, {high_sse_share:.1f}% of squared error",
             ha="center", fontsize=9, color="#3D4651")
    save_figure(fig, "figure1_ensemble_error_analysis", svg=True)

    by_id = {r["imageid"]: r for r in rows}
    case_labels = ["Near-accurate", "Severe underestimate", "Overestimate", "Partial tree occlusion"]
    case_records = []
    fig = plt.figure(figsize=(8.1, 7.75))
    outer = fig.add_gridspec(2, 2, left=0.04, right=0.96, bottom=0.075,
                             top=0.95, wspace=0.09, hspace=0.18)
    for i, (image_id, case_label) in enumerate(zip(CASE_IDS, case_labels)):
        row = by_id[image_id]
        actual, estimate = float(row["price"]), float(row["pred"])
        signed = estimate - actual
        source_image = ROOT / "house_dataset/train" / image_id
        with Image.open(source_image) as original:
            pixels = np.array(original.convert("RGB"))
        panel = outer[i // 2, i % 2].subgridspec(2, 1, height_ratios=[1, 0.18], hspace=0.02)
        ax = fig.add_subplot(panel[0])
        ax.imshow(pixels, interpolation="none")
        ax.set_axis_off()
        ax.set_title(f"({chr(97+i)}) {image_id} | {case_label}", loc="left", fontsize=10, pad=7)
        text_ax = fig.add_subplot(panel[1])
        text_ax.set_axis_off()
        text_ax.text(0, 0.88, f"True: {actual:,.2f}    Predicted: {estimate:,.2f}",
                     transform=text_ax.transAxes, fontsize=10, va="top")
        text_ax.text(0, 0.40, f"Error (prediction - true): {signed:+,.2f}",
                     transform=text_ax.transAxes, fontsize=9.5, va="top",
                     color=GREY if abs(signed) < 1 else (RED if signed < 0 else ORANGE))
        case_records.append({"imageid": image_id, "true_price": actual, "prediction": estimate,
                             "error_prediction_minus_true": signed, "fold": int(row["fold"]),
                             "source_image": str(source_image.relative_to(ROOT)),
                             "image_sha256": digest(source_image)})
    fig.text(0.5, 0.028, "Nine-predictor NNLS blend | OOF predictions | All values in US$1,000",
             ha="center", fontsize=10, color="#3D4651")
    save_figure(fig, "figure2_ensemble_cases")

    metadata = {
        "model": rows[0]["model"], "prediction_source": str(SOURCE.relative_to(ROOT)),
        "prediction_source_sha256": digest(SOURCE), "n": len(rows),
        "prices_unit": "US$1,000", "mse_unit": "(US$1,000)^2",
        "error_definition": "prediction - true price",
        "mse_recomputed_from_rounded_csv": float(squared.mean()),
        "mae_recomputed_from_rounded_csv": float(np.abs(error).mean()),
        "price_band_statistics": bands,
        "calibration_bins": bins,
        "calibration_bin_method": "Stable sort by prediction; 10 consecutive groups of 800; input row order breaks ties.",
        "high_price_summary": {"threshold_inclusive": 1000, "n": high_n,
                               "sample_share_percent": high_share,
                               "squared_error_share_percent": high_sse_share},
        "cases": case_records,
        "evaluation_note": "Development OOF estimates; member selection and the whole blending pipeline were not independently validated.",
        "rounding_note": "Per-image CSV prices/predictions have 3 decimal places; aggregate MSE differs slightly from final_ensemble.csv.",
        "case_selection_note": "Use the four image IDs designated by the user in the project case-selection screenshot; no new selection rule inferred.",
    }
    (OUT / "figure_data.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved two report figures and underlying statistics to {OUT}")


if __name__ == "__main__":
    main()
