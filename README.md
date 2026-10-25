# COMP90086 Final Project — House Prices from Exterior Photos

Predicting a house's sale price (in $1000 USD) from a single exterior listing
photograph, using frozen self-supervised vision features with lightweight
regression heads, combined by a non-negative least-squares blend.

Hongze Li (1820462) and Jiatong Wu (1822875).

Full experiment log: [`outputs/experiments/RESULTS.md`](outputs/experiments/RESULTS.md).
Narrative of how the experiments led into one another (Chinese):
[`outputs/experiments/NARRATIVE.zh.md`](outputs/experiments/NARRATIVE.zh.md).

---

## Results

Out-of-fold MSE on the frozen five-fold split (`data/splits/folds.csv`), raw
price in $1000 USD, lower is better.

| Configuration | OOF MSE | R² |
| --- | ---: | ---: |
| Predict the training mean (baseline) | 156,686 | 0 |
| Ridge on frozen DINOv2 ViT-B/14 @224 | 88,663 | 0.434 |
| MLP on frozen DINOv2 ViT-L/14 @336 | 76,807 | 0.510 |
| MLP on frozen DINOv3 ViT-H+/16 @336 | 70,818 | 0.548 |
| MLP on six concatenated feature groups | 69,175 | 0.556 |
| …same, predictions averaged over three seeds | 68,522 | 0.563 |
| **Nine-member NNLS blend + linear recalibration** | **65,278** | **0.583** |

Kaggle (test set, held out):

| Submission | Score | Note |
| --- | ---: | --- |
| ResNet-50 + ConvNeXt-Tiny, 2026-09-23 | 100,376.08 | first baseline |
| DINOv2 four-way concatenation, 2026-10-13 | 70,096.88 | |
| **Six-group concatenation, three seeds, 2026-10-20** | **63,952.39** | **last submission before the leaderboard closed** |
| Nine-member blend | 62,127.62 | submitted after the leaderboard closed; **does not count** |

The nine-member blend is the best configuration by out-of-fold error, but it
was finalised after the competition closed, so 63,952.39 is the scored result.

**Out-of-fold estimates are conservative, not optimistic.** 68,522 OOF against
63,952.39 on test is 7.1% higher. Each out-of-fold model trains on 6,400
images; the submitted model is refitted on all 8,000.

---

## Setup

Python 3.12. **PyTorch must come from the CUDA index**, or pip silently
installs a CPU-only build.

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows; source .venv/bin/activate elsewhere
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt   # exact pinned versions in requirements.lock.txt

python -c "import torch; print(torch.cuda.is_available())"   # must print True
```

Developed on a single 8 GB GPU (RTX 4060 Ti). Every frozen-feature experiment
fits in that budget; the largest backbone used, DINOv3 ViT-H+/16 at 336 px,
peaks at about 5.3 GB for inference.

## Data

**The images are not in this repository and are not in the code submission.**
The code expects them under `data/raw/house_dataset/`:

```
data/raw/house_dataset/
    train/*.jpg          8,000 images, ids 1-8000
    test/*.jpg           3,000 images, ids 8001-11000
    train.csv            imageid, price
    test.csv             imageid
    sample_solution.csv  imageid, price
```

To fetch them, join the competition on Kaggle, accept the rules, then:

```bash
python src/download_data.py --slug comp-90086-2026
python src/inspect_data.py --root data/raw/house_dataset
```

`data/splits/folds.csv` is committed, so the exact split used for every number
above is reproducible without rerunning the duplicate scan.

## Reproducing the final result

```bash
# 1. Frozen folds — already committed; only rerun to rebuild the split
python src/make_splits.py --train-csv data/raw/house_dataset/train.csv \
                          --image-dir data/raw/house_dataset/train

# 2. Cache the nine blend members' features (minutes in total on one GPU)
for SPLIT in train test; do
  python src/extract_features.py --split $SPLIT --crop-mode squash \
      --model vit_huge_plus_patch16_dinov3.lvd1689m --img-size 336 --batch-size 4
  python src/extract_features.py --split $SPLIT --crop-mode squash \
      --model vit_huge_plus_patch16_dinov3.lvd1689m --img-size 256 --batch-size 8
  python src/extract_features.py --split $SPLIT --crop-mode squash \
      --model vit_huge_plus_patch16_dinov3.lvd1689m --img-size 224 --batch-size 8
  python src/extract_features.py --split $SPLIT --crop-mode squash \
      --model vit_large_patch16_dinov3.lvd1689m --img-size 336 --batch-size 8
  python src/extract_features.py --split $SPLIT --crop-mode squash \
      --model vit_large_patch16_dinov3.lvd1689m --img-size 224 --tiles 2 --batch-size 8
  python src/extract_features.py --split $SPLIT --crop-mode squash \
      --model vit_large_patch14_reg4_dinov2.lvd142m --img-size 336 --batch-size 8
  python src/extract_features.py --split $SPLIT --crop-mode squash \
      --model vit_large_patch14_reg4_dinov2.lvd142m --img-size 518 --batch-size 4
  python src/extract_meta.py --split $SPLIT
done

# 3. Final model: nine members, NNLS blend, linear recalibration, submission CSV
python src/make_ensemble_submission.py
# -> outputs/submissions/final_ensemble.csv   (OOF MSE 65,278)

# 4. The scored submission (six-group concatenation, three seeds)
python src/exp_heads.py --heads mlp_mse --seeds 0 1 2 --write-submission \
    --out final_best --tags \
    "vit_huge_plus_patch16_dinov3_336_squash+vit_huge_plus_patch16_dinov3_256_squash+vit_large_patch16_dinov3_336_squash+vit_large_patch14_reg4_dinov2_336_squash+vit_large_patch16_dinov3_224_squash_tile2+photometa"
# -> outputs/submissions/final_best_mlp_mse.csv   (OOF 68,522, Kaggle 63,952.39)
```

Nothing in this repository uploads to Kaggle; `--write-submission` only writes
a local CSV.

## Reproducing the reported experiments

```bash
python src/exp_ridge.py                    # E1  aspect ratio x target decoding
python src/exp_heads.py --heads ridge xgboost lightgbm mlp_mse mlp_binned mlp_ordinal
python src/exp_objective.py --seeds 0      # E5  price-dependent loss weighting
python src/exp_tiered.py --seeds 0         # E6  tier routing, incl. the oracle bound
python src/exp_blend.py --top 12           # E13 prediction-level fusion
python src/export_oof_predictions.py       # per-image OOF table for error analysis
python src/make_figures.py                 # report figures 1-4
python src/figures_part2.py                # report figures 5-7
```

Two gotchas that cost us time: run scripts **without** `python -I` (it drops
`src/` from the import path), and never pipe a long job through `tail` — log to
a file instead, or the run is blind until it exits.

## Layout

| Path | Purpose |
| --- | --- |
| `src/download_data.py` | Fetch and unpack the competition archive |
| `src/inspect_data.py` | Integrity survey, price distribution, baseline MSE |
| `src/make_splits.py` | pHash duplicate grouping + price-stratified GroupKFold |
| `src/extract_features.py` | Frozen-backbone feature cache (CLS + mean patch; `--tiles` for the 2×2 grid) |
| `src/extract_meta.py` | 19 low-level photographic statistics |
| `src/common.py` | Metrics, calibration diagnostics, target decoders, submission writer |
| `src/bins.py` | Price discretisation for the distribution head |
| `src/exp_ridge.py` | E1 — aspect ratio × target decoding |
| `src/exp_heads.py` | E3/E4/E11/E12 — features × heads, seed repeats |
| `src/exp_objective.py` | E5 — target scale and price-dependent weighting |
| `src/exp_tiered.py` | E6 — low/mid/high routing and the oracle bound |
| `src/exp_blend.py` | E13 — prediction-level fusion over stored OOF vectors |
| `src/make_ensemble_submission.py` | Final nine-member blend and its submission CSV |
| `src/export_oof_predictions.py` | Per-image OOF table used for the error-analysis figures |
| `src/train_finetune.py` | End-to-end fine-tuning (not used in the final model; see Limitations) |
| `src/make_figures.py`, `src/figures_part2.py` | Report figures |
| `outputs/experiments/` | Metric tables, the experiment log, per-image predictions |
| `outputs/submissions/` | Kaggle-format prediction files |

`data/`, `features/` and model weights are gitignored. `data/splits/folds.csv`
and the small metric CSVs are tracked, because they are what makes the reported
numbers reproducible.

---

## Findings worth knowing before changing anything

**Tail under-prediction is MSE-optimal shrinkage, not a bug.** Binned by true
price the model looks badly biased (−532 in the $1.5–2M band, which carries 39%
of all squared error). Binned by *prediction* it is calibrated: all ten deciles
sit on the diagonal, slope 0.996. Conditioning on the true price makes any
predictor with limited information look biased. Inflating the high end moves
predictions away from the conditional mean and increases MSE; only a more
discriminative representation helps.

**Pretraining objective and architecture each matter about 11%; capacity
almost not at all.** Each pair below differs in exactly one factor:

| Change | Effect on OOF MSE |
| --- | ---: |
| ImageNet-22k supervised → DINOv3 self-supervised (ConvNeXt-B) | −11.1% |
| ConvNeXt-L → ViT-L (both DINOv3) | −11.2% |
| ConvNeXt-B → ConvNeXt-L (both DINOv3) | −1.5% |

Every image here is the same object class, so the invariances that help a
supervised classifier separate categories also discard the within-class
appearance variation price depends on.

**Optimal input resolution is set by the source image, not the backbone.**
DINOv2 (native 518) and DINOv3 (native 256) both optimise near 336, which
matches the 400×300 source photographs. DINOv2 is worse at its own native
resolution than at 336.

**Ensemble weight does not track member accuracy.** The two heaviest members of
the blend are among the weakest individually; they are the only member from a
different pretraining family and the only one with a different input geometry.
Diversity beats individual accuracy here.

**Shortcuts were checked and ruled out.** Image ids are uncorrelated with price
(Pearson −0.006); 0.37% of test images have a train near-duplicate at pHash
distance ≤6; k-NN retrieval on the same features reaches only 98,839 against
70,818 for an MLP, with k=1 at R² 0.035.

**Negative results that are in the report on purpose.** A binned-distribution
head loses 10–14% at 32–64 bins and is indistinguishable from a scalar head at
8–16; an ordinal auxiliary task is neutral at λ≤0.3 and harmful at λ=1; tier
routing costs 14.5% (soft) to 26.3% (hard) despite a 62.1% gate accuracy.
Each was swept and measured against seed variance before being set aside.

**Any gap below about 1,000 MSE is not a result.** Seed standard deviation is
±130–450 depending on the feature set; several apparent wins reversed once
predictions were averaged over three seeds.

## Limitations

**End-to-end fine-tuning was not adequately evaluated.** Two attempts were
made. The first (DINOv2 ViT-B @280, 20 epochs) returned 73,481 on fold 0 behind
the frozen-feature concatenation's 69,827 on the same fold, but its console
output was lost to buffering so under-training could not be distinguished from
a failed run. The second (DINOv3 ViT-L @256, 40 epochs, mixup, layer-wise LR
decay) was stopped after 135 minutes: one fold costs about two hours on the
available GPU and five folds would cost over ten. `src/train_finetune.py` is
included and works, but no fine-tuned model contributes to the reported result.

This is a gap rather than a negative result. The frozen-feature route was
chosen deliberately: cached features make an experiment loop that runs in
seconds, which is what allowed seventy-plus controlled configurations to be
compared, where the same compute would have bought two or three fine-tuning
runs. Given the oracle result in E6 — routing by the true price tier reaches
40,563 — coarse tier separability is where the headroom is, and systematic
fine-tuning plus larger ensembles are the obvious next steps.
