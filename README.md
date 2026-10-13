# COMP90086 Final Project — House Prices from Exterior Photos

Predicting a house's sale price (in $1000 USD) from one exterior listing
photograph. This repo holds Jiatong's experimental line. Hongze is running a
separate line; the plan is to compare and then ensemble.

**Everything below is out-of-fold MSE on the frozen 5-fold split**
(`data/splits/folds.csv`), raw price, lower is better. The full experiment log
with every table lives in [`outputs/experiments/RESULTS.md`](outputs/experiments/RESULTS.md).

---

## Where we are

| | MSE | R² |
| --- | ---: | ---: |
| Predict the training mean (baseline) | 156,686 | 0 |
| Our 2026-09-23 submission (public LB) | 100,376 | — |
| Ridge on frozen DINOv2 ViT-B/14 @224 | 88,663 | 0.434 |
| MLP on frozen DINOv2 ViT-L/14 @224 | 78,078 | 0.502 |
| MLP on frozen ViT-L/14 @336 | 76,807 | 0.510 |
| **MLP on 4-way concatenated frozen features** | **75,218** | **0.520** |
| *Target* | *~50,000* | *0.681* |
| *Public leaderboard top (2026-10-13)* | *42,554* | *0.729* |

The frozen-feature line is close to exhausted. A fine-tuning pilot is running;
everything after this point depends on how steep that curve is.

**No Kaggle submission has been made from this repo.** One CSV is sitting in
`outputs/submissions/` waiting for a decision. Note the daily submission limit
— the brief treats exceeding it as misconduct.

**Local validation tracks the leaderboard.** On the September submission,
local 101,072 vs public 100,376 — 0.7% apart. So we can select models locally
and do not need to burn submissions probing.

---

## Things that change your plan, Hongze

Please read these two before spending more time on your branch.

### 1. There is no high-resolution branch. The images cap at 400×300.

Measured over all 11,000 images: max width 400, max height 300, short side
median 277, **zero images with a short side ≥384**. Any "higher-resolution
whole image" condition is pure upsampling and cannot add information. The
whole+local-patch idea is also weakened: a one-third crop is about 133×92 px.

This does not mean resolution is worthless — 224→280→336 does improve things
(−2.2%, then −0.7%) — but the gain comes from finer ViT tokenisation (256 vs
576 patches), not from new image content. That distinction is worth a
paragraph in the report, and it is a more interesting finding than the
comparison we originally planned.

### 2. Please confirm SigLIP 2 on Ed before building on it.

It is an image–text dual encoder, not a generative MLLM, so it is very
probably fine. But rule 3 in the brief ("no pretrained MLLMs") is vague and
the penalty is zero marks. Not worth the risk unconfirmed. This line
deliberately uses only pure-vision pretraining (DINOv2 self-supervised,
ConvNeXt ImageNet-22k) so it is unambiguous either way.

---

## The two findings that should shape the report

### High-price under-prediction is not a bug. It is MSE-optimal shrinkage.

Binned by **true** price, the model looks badly biased — houses at
$1.5M–2M are under-predicted by 661 on average, and that band alone carries
41% of all squared error.

Binned by **prediction**, it is almost perfectly calibrated — all ten deciles
sit on the diagonal (largest gap 14, calibration slope 0.996). Of the houses
this model values at 1259, the average sale price is 1259.

Conditioning on the true price makes *any* predictor look biased, because
among houses that happen to be expensive, the ones the model cannot identify
drag its average prediction down. That is regression toward the mean, and
under limited signal it is the correct response. **Inflating high predictions
would increase MSE.** The only legitimate route to a smaller tail error is a
more discriminative representation.

Our submitted plan framed this as an error to fix. It is not, and the report
should say so with both tables side by side — it is a better result than the
one we set out to get.

### Pretraining objective matters more than architecture here.

ConvNeXt-Base (ImageNet-22k, supervised) scores R² 0.370 on its own — 16%
*worse* than the much smaller DINOv2 ViT-B. Plausible reason: every image in
this dataset is the same object class, and the invariances that make a
classifier good at telling a house from a car are exactly the ones that
discard within-class appearance variation — which is where all the price
signal lives. A self-supervised objective has no such pressure.

Practical consequence: stay inside the DINOv2 family for the backbone, and
treat supervised-ImageNet backbones as ensemble diversity at best. Adding
ConvNeXt to ViT-L/14 @336 actually *hurt* (77,937 vs 76,807); it only helped
inside a 4-way concatenation.

---

## Lessons (including two hypotheses that failed)

Both of these were my calls, both were wrong, and both stay in the report
because a cleanly-tested refuted hypothesis is worth more marks than a
cherry-picked winner.

**1. "Target scale is the biggest lever." It is not — it is worth 0.25%.**

The theory was right in direction: fitting on log price and decoding with a
naive `exp(µ)` estimates the conditional *median* and under-predicts, and the
measured bias was −49.3, exactly as predicted. But correcting it only matches
plain raw-price regression, because the target here is bounded [195, 2000]
with skew 1.26 (log skew 0.03) — raw MSE was never unstable, so the problem
the log transform solves does not exist on this dataset. Lesson: check whether
the *precondition* of a textbook fix holds before prioritising it.

**2. The binned-distribution head — proposed as the main method — gave nothing.**

The idea: predict a distribution over log-spaced price bins with soft-target
cross-entropy, decode the expectation. It directly estimates E[price|x] (the
MSE-optimal predictor), bounds the per-sample gradient against the heavy tail,
and yields free predictive uncertainty. It is also a strict generalisation of
the ordinal auxiliary loss in our project plan.

Swept over 8–64 bins, two smoothing widths, and a hybrid `CE + λ·MSE(mean)`
loss, then measured over 3 seeds:

* 32–64 bins lose 10–14% — robustly, across every variant.
* 8–16 bins are statistically indistinguishable from a plain scalar head
  (76,659 ± 412 for the scalar vs 77,222 ± 456 for 12 bins).
* The hybrid loss does **not** rescue it, which rules out "objective doesn't
  match the metric" as the explanation.

So the cost is the estimation variance of fitting 64 outputs instead of 1 from
6,400 rows at R² ≈ 0.5. The precedents (DEX for age, AdaBins for depth)
operate at far higher signal-to-noise. We use the scalar head.

**3. Measure seed variance before believing a small margin.**

At one seed the 12-bin head "beat" the scalar head by 236 MSE. Seed std turned
out to be ±412. The margin was noise, and the ranking flipped once predictions
were averaged over 3 seeds. Any gap under ~1,000 MSE in this project is not a
result yet.

**4. Averaging 3 seeds is worth 1.8% for free** (78,078 → 76,659). MSE rewards
variance reduction directly, so the final model must be an ensemble. This is
cheaper than almost any modelling improvement.

**5. Check the data before designing the method.** The resolution ceiling and
the price clipping were both found in the first hour and both invalidated
parts of the submitted plan. Clipping predictions to [195, 2000] is free MSE.

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

Images are **not** in this repo. Join the competition on Kaggle, accept the
rules, then:

```bash
python src/download_data.py --slug comp-90086-2026   # unpacks to data/raw/
python src/inspect_data.py --root data/raw/house_dataset
```

Expected layout: `data/raw/house_dataset/{train,test}/*.jpg` plus
`train.csv`, `test.csv`, `sample_solution.csv`.

## Reproducing the results

```bash
# 1. Frozen folds (already committed; only rerun if the split must change)
python src/make_splits.py --train-csv data/raw/house_dataset/train.csv \
                          --image-dir data/raw/house_dataset/train

# 2. Cache frozen features (seconds each; 11k images)
python src/extract_features.py --split train --crop-mode squash \
    --model vit_large_patch14_reg4_dinov2.lvd142m --img-size 336 --batch-size 8
python src/extract_features.py --split test  --crop-mode squash \
    --model vit_large_patch14_reg4_dinov2.lvd142m --img-size 336 --batch-size 8

# 3. E1: aspect-ratio treatment x target decoding
python src/exp_ridge.py --write-submission

# 4. E3/E4: features and heads ('+' concatenates cached feature sets)
python src/exp_heads.py --heads ridge lightgbm mlp_mse mlp_binned
python src/exp_heads.py --heads mlp_mse --seeds 0 1 2 \
    --tags vit_large_patch14_reg4_dinov2_224_squash+vit_large_patch14_reg4_dinov2_336_squash

# 5. Fine-tuning
python -u src/train_finetune.py --folds 0 --epochs 20 --batch-size 16
```

Two gotchas that cost time: run scripts **without** `python -I` (it drops
`src/` from the import path), and pipe through `grep --line-buffered` or you
will see no output until the job ends.

`--write-submission` only writes a local CSV. Nothing in this repo uploads to
Kaggle.

## Layout

| Path | Purpose |
| --- | --- |
| `src/inspect_data.py` | Integrity survey, price distribution, baseline MSE |
| `src/make_splits.py` | pHash duplicate grouping + price-stratified GroupKFold |
| `src/extract_features.py` | Frozen-backbone feature cache (CLS + mean patch) |
| `src/common.py` | Metrics, calibration diagnostics, target decoders, submission writer |
| `src/bins.py` | Price discretisation for the distribution head |
| `src/exp_ridge.py` | E1 — aspect ratio × target decoding |
| `src/exp_heads.py` | E3/E4 — features × heads, with seed repeats |
| `src/train_finetune.py` | End-to-end fine-tuning, 5-fold, EMA, layer-wise LR decay |
| `outputs/experiments/RESULTS.md` | Full experiment log |

`data/`, `features/` and `outputs/` are gitignored — images must never be
committed, and the brief says not to ship them in the code submission either.

---

## Next steps

| Step | Expected | Status |
| --- | --- | --- |
| Frozen features | 75,218 | done, near its ceiling |
| Single fine-tuned model | 60–65k? | pilot running (fold 0, 20 epochs) |
| Fine-tuned + 5 folds + multi-seed | 55–58k | blocked on the pilot |
| Multi-backbone + TTA ensemble | 50–54k | blocked |

**Known engineering problem.** GPU utilisation during fine-tuning sits at
30–64%: augmentation runs in the main process because on Windows
`num_workers>0` spawns workers that each copy the 2.9 GB in-RAM image array.
Before the 5-fold run this needs fixing — most likely by returning uint8 crops
from the dataset and doing colour jitter and normalisation batched on the GPU.
At 5 folds × 20 epochs the wasted time is hours.

### Open questions for us to decide together

1. **Submit the 75,218 model now?** It would improve on our September 100,376
   and confirms the submission path works. Costs one of the daily slots.
2. **How do we split the remaining work** now that we are running separate
   lines? Suggestion: Jiatong takes fine-tuning and the ensemble, Hongze takes
   a genuinely different representation (not SigLIP until Ed confirms) so the
   ensemble members decorrelate. Error analysis and the report are joint.
3. **Report start date.** Four pages of two-column IEEE is routinely
   underestimated. The Method and Related Work sections can be drafted now —
   the findings above are already stable enough to write up.
4. **The 3090.** The local 4060 Ti (8 GB) handles everything so far. We need it
   mainly for the final multi-seed 5-fold runs. Worth lining up now.
