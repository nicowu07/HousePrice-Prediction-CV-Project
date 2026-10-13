# Experiment log

All numbers are out-of-fold MSE on the frozen 5-fold split
(`data/splits/folds.csv`), raw price in $1000 USD. Lower is better.

**Reference points**

| Reference | MSE | Note |
| --- | --- | --- |
| Predict the training mean | 156,686 | the constant-prediction baseline; MSE = price variance |
| Our 2026-09-23 submission | 100,376 | public leaderboard, ResNet-50 + ConvNeXt-Tiny 50:50 |
| Current leaderboard top | 42,554 | public leaderboard, 2026-10-13 |

Local validation tracked the leaderboard to within 0.7% on the September
submission (local 101,072 vs public 100,376), so out-of-fold MSE is treated
as a trustworthy proxy and model selection never consults the leaderboard.

---

## Data integrity

Checked before any modelling, because each of these would invalidate
downstream comparisons if left unexamined.

| Check | Result |
| --- | --- |
| train.csv ↔ train/ images | 8,000 ↔ 8,000, no missing or extra, ids unique |
| test.csv ↔ sample_solution.csv | 3,000 rows, identical ids **in identical order** |
| train / test id ranges | 1–8000 and 8001–11000, no overlap |
| Near-duplicates within train | 9 pairs at pHash Hamming ≤ 6 → 7,991 groups; kept within a fold |
| Near-duplicates train ↔ test | 11 of 3,000 test images (0.37%) at Hamming ≤ 6 — negligible, no exploitable leak |
| Image id vs price | Pearson −0.006, Spearman −0.002; id-adjacent pairs differ by 417 vs 429 for random pairs → ids carry no signal |
| Prices | no NaN, none ≤ 0; range [195, 2000]; 1,815 distinct values; 77.6% integral, clustered on listing price points (799, 699, 749, …) |

**Consequences for the method.** Prices are filtered to a closed interval, so
every prediction is clipped to [195, 2000] — a prediction outside it can only
increase squared error. The expensive tail dominates the metric: prices ≥1000
are 16.0% of rows but 64.9% of total squared error (the project plan estimated
15.8% / 55.4% from a smaller sample, so the premise holds and is slightly
stronger than assumed).

**Resolution ceiling.** No image exceeds 400×300; the short side has median
277 and maximum 300, and *no* image has a short side ≥384. Any "high
resolution" branch is therefore pure upsampling and cannot add information.

---

## E1 — aspect-ratio treatment and target decoding (Ridge, frozen DINOv2 ViT-B/14 @224)

The metric is MSE on raw price, whose Bayes-optimal predictor is E[price|x].
Fitting on log price is numerically convenient but `exp` of the fitted value
estimates the conditional *median*, which is systematically low; each decoder
is a different route back to the mean. Decoders are fitted on inner
out-of-fold predictions, never in-sample residuals.

| Aspect | Decoder | MSE | R² | bias | slope |
| --- | --- | ---: | ---: | ---: | ---: |
| squash | isotonic | **88,663** | 0.434 | +0.6 | 0.996 |
| squash | raw price | 88,883 | 0.433 | +0.9 | 0.995 |
| squash | log-normal `exp(µ+σ²/2)` | 89,147 | 0.431 | +2.3 | 0.955 |
| squash | Duan smearing | 89,157 | 0.431 | +3.4 | 0.953 |
| center | raw price | 90,921 | 0.420 | +0.6 | 0.995 |
| squash | naive `exp(µ)` | 91,526 | 0.416 | **−49.3** | 1.027 |
| pad | naive `exp(µ)` | 95,098 | 0.393 | −50.8 | 1.020 |

**Aspect ratio: squash wins** (−2.5% vs centre crop, −4.2% vs letterbox).
Centre-cropping a 4:3 listing photo discards ~31% of its width, which is where
the driveway, garage and planting live.

**Target scale: the predicted effect is real but small.** Naive `exp(µ)` shows
exactly the predicted bias (−49.3), and correcting it recovers ~3%. But plain
raw-price regression is already within 0.25% of the best decoder, because the
bounded, only mildly skewed target (skew 1.26; log skew 0.03) never made raw
MSE unstable in the first place. The theoretical argument holds; its practical
weight on *this* dataset does not justify treating it as a primary lever.

---

## E4 — head, on identical frozen features

| Head | MSE (ViT-B/14 @224) | R² |
| --- | ---: | ---: |
| MLP + MSE | **84,872** | 0.458 |
| Ridge | 88,704 | 0.434 |
| LightGBM | 91,812 | 0.414 |
| MLP + binned distribution | 94,833 | 0.395 |

LightGBM losing to Ridge is consistent with the input: no single dimension of
a dense ViT embedding is individually meaningful, so axis-aligned splits are a
poor inductive fit.

### The binned-distribution head: the cost is output dimensionality

It was proposed as the main method: predict a distribution over log-spaced
price bins with soft-target cross-entropy and decode the expectation, which
estimates E[price|x] directly, bounds the per-sample gradient against the
heavy tail, and yields predictive uncertainty for free. A sweep on ViT-L/14
@224 features (point-regression reference **78,078**, single seed) locates
exactly where it costs:

| Variant | MSE |
| --- | ---: |
| 12 bins, σ=3.0 | **77,842** |
| 8 bins, σ=3.0 | 78,289 |
| 16 bins, σ=3.0 | 78,826 |
| 24 bins, σ=3.0 | 79,936 |
| 32 bins, σ=3.0, +λ·MSE (λ=2) | 81,408 |
| 64 bins, σ=3.0 | 86,156 |
| 32 bins, σ=1.5 | 86,845 |
| 64 bins, σ=1.5, +λ·MSE (λ=5) | 88,517 |
| 64 bins, σ=1.5, +λ·MSE (λ=1) | 88,920 |

Performance improves monotonically as the output is coarsened, reaching a
shallow optimum around 12 bins. Two readings are warranted, and they differ
in strength:

* **Fine binning is robustly worse.** 32–64 bins lose 10–14% across several
  smoothing widths, and adding an explicit squared-error term on the decoded
  mean does not rescue them. The failure is therefore not a mismatch between
  objective and metric; it is the estimation variance of fitting 64 outputs
  instead of 1 from 6,400 rows at R² ≈ 0.5. The precedents for this head (DEX
  for age, AdaBins for depth) operate at far higher signal-to-noise.
* **Coarse binning is indistinguishable from point regression.** The 12-bin
  configuration edges out the scalar head by 0.3%, which is not a meaningful
  margin from a single seed. Seed-variance measurement is pending before any
  claim is made in either direction.

---

## E3 — features (MLP + MSE head held constant)

| Features | dim | MSE | R² |
| --- | ---: | ---: | ---: |
| DINOv2 ViT-L/14 @224 | 2048 | **78,078** | 0.502 |
| DINOv2 ViT-B/14 @336 | 1536 | 82,414 | 0.474 |
| DINOv2 ViT-B/14 @280 | 1536 | 82,966 | 0.470 |
| DINOv2 ViT-B/14 @280 + flip-TTA | 1536 | 83,223 | 0.469 |
| DINOv2 ViT-B/14 @224 | 1536 | 84,872 | 0.458 |

**Backbone capacity beats resolution**: ViT-L @224 is 5.3% better than ViT-B
@336 and cheaper to extract.

**Resolution still helps past the native ceiling.** 224→280 gives −2.2%,
280→336 a further −0.7%, even though 336 upsamples every image (native short
side median 277). Upsampling adds no high-frequency information, so the gain
must come from finer spatial tokenisation — 256 patches at 224 versus 576 at
336 — which separates "resolution" from "information content".

**Flip-TTA on features slightly hurts** (83,223 vs 82,966). Averaging a photo
with its mirror destroys orientation cues (which side the driveway is on,
shadow direction, which face is lit) that the backbone does encode.

---

## Calibration: the tail bias is not a defect

Binned by **true** price, the best model looks badly biased:

| Band | n | % rows | MSE | bias | % of total SE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 195–400 | 1685 | 21.1 | 40,177 | +151 | 10.8 |
| 400–700 | 3031 | 37.9 | 37,518 | +91 | 18.2 |
| 700–1000 | 2005 | 25.1 | 47,035 | −12 | 13.9 |
| 1000–1500 | 735 | 9.2 | 141,611 | −265 | 15.3 |
| 1500–2000 | 544 | 6.8 | 518,411 | −661 | 41.5 |

Binned by **prediction**, it is almost perfectly calibrated:

| Pred. decile | mean pred | mean true | gap | RMSE |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 341.9 | 343.2 | +1.3 | 152.1 |
| 5 | 635.6 | 635.9 | +0.3 | 251.2 |
| 8 | 883.0 | 897.1 | +14.1 | 355.5 |
| 10 | 1259.1 | 1259.2 | +0.1 | 409.3 |

Conditioning on the true price makes even a Bayes-optimal predictor look
biased, because among houses that *happen* to be expensive, the ones the model
cannot identify drag its average prediction down. Conditioning on the
prediction asks the question that matters, and the answer is that the
shrinkage is correct: the calibration slope is 0.996 and every decile sits on
the diagonal. **Inflating high predictions would increase MSE.** The only
legitimate route to a smaller tail error is a more discriminative
representation, not a post-hoc correction.

Secondary observation: RMSE rises monotonically with predicted price
(152 → 409), i.e. the problem is heteroscedastic — expensive houses are
intrinsically harder to price from a photograph.

---

## Revision to the project plan

The submitted plan framed high-price under-prediction as an error to fix and
proposed an ordinal auxiliary loss to fix it, plus a high-resolution input
branch. The data refute both premises:

* there is no higher resolution to use (400×300 ceiling), and
* the under-prediction is MSE-optimal shrinkage, not a bug.

The distributional head (a strict generalisation of the planned ordinal
auxiliary task) was implemented and tested anyway, and lost under a
hyperparameter sweep. Effort moved instead to representation quality, which
the evidence identifies as the binding constraint.
