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

---

## E5 — training objective and price-dependent weighting

The metric is MSE on raw price; the baseline MLP trains on squared error in
log price. Those differ, and the difference has a closed form. With
`p_hat = exp(mu_hat)` and `p = exp(z)`,

    (p_hat - p)^2 = e^{2z}(e^{mu_hat - z} - 1)^2  ~=  p^2 (mu_hat - z)^2

to first order. **Raw-price MSE is log-space MSE weighted by p².** So
switching the target to raw price and up-weighting expensive houses are two
corrections for the same mismatch, not two independent ideas. That lets a
single parameter sweep subsume both: `w ∝ p^α`, with α = 0 the current
baseline and α = 2 the first-order equivalent of raw MSE.

Loss weighting is used rather than resampling: the expected gradient is the
same, but resampling adds sampling noise and changes the price composition of
each batch, which perturbs the BatchNorm statistics and would put a second
variable into the comparison. Weights are normalised to mean 1 so that
"weighting" is not confounded with "a higher learning rate".

Single seed, 4-way concatenated features:

| Condition | MSE | vs A | n_eff | n_eff / n |
| --- | ---: | ---: | ---: | ---: |
| α = 0.5 | 75,106 | −0.15% | 7,467 | 93% |
| **A — log target, unweighted** | **75,218** | — | 8,000 | 100% |
| α = 1.0 | 75,265 | +0.06% | 6,146 | 77% |
| α = 1.5 | 75,350 | +0.18% | 4,631 | 58% |
| C — 2× for price ≥ 1000 | 75,430 | +0.28% | 7,274 | 91% |
| B — raw-price target | 75,806 | +0.78% | 8,000 | — |
| α = 2.0 | 75,867 | +0.86% | 3,360 | 42% |
| α = 3.0 | 80,399 | +6.89% | 1,839 | 23% |

Three seeds, leading conditions (± is the standard deviation across seeds):

| Condition | MSE | ± | vs A |
| --- | ---: | ---: | ---: |
| α = 0.5 | 74,266 | 161 | −0.27% |
| C — 2× for price ≥ 1000 | 74,345 | 70 | −0.16% |
| A — baseline | 74,467 | 129 | — |
| B — raw-price target | 74,860 | 150 | +0.53% |

**B and α = 2.0 land 61 MSE apart** (75,806 vs 75,867), which is well inside
seed noise. Two independently implemented routes converging on the same number
is a direct empirical confirmation of the equivalence derived above.

**The metric-aligned weight is not the best weight.** Theory says α = 2;
the empirical curve is flat to α ≈ 0.5 and then climbs. The reason is
estimation efficiency: weighting reduces the effective sample size
`n_eff = (Σw)² / Σw²`, and α = 2 discards 58% of it. Two forces act in
opposite directions — metric alignment pulls α up, estimation efficiency pulls
it down — and with R² ≈ 0.52 the variance term dominates, so the optimum sits
near the efficiency end. The MSE curve turns upward exactly where n_eff starts
to collapse (Figure 3).

Practically: mild weighting (α ≈ 0.5, or the 2× heuristic) is neutral to
very slightly positive and within seed noise; strong weighting and the
raw-price target are measurably worse. The baseline objective is kept.

---

## E6 — does splitting by price tier help?

Tertile tiers, three MLP experts, and a three-way gate on the same features.
Single seed.

| Routing | MSE | vs single | R² |
| --- | ---: | ---: | ---: |
| oracle (routed by the **true** tier) | 40,563 | −46.07% | 0.741 |
| single model | 75,218 | — | 0.520 |
| soft (probability-weighted mixture) | 86,157 | +14.54% | 0.450 |
| hard (routed by the predicted tier) | 95,028 | +26.34% | 0.394 |

Gate accuracy is 62.1% against a 33.4% majority-class baseline — far better
than chance, far short of usable. Errors are mostly to an adjacent tier, but
415 of 8,000 jump two tiers.

**Realisable routing is clearly harmful.** Soft beats hard, as expected, since
it never commits to a single wrong tier — but both lose badly to one model. Two
reasons: each expert sees a third of the data, and its output range is
confined to its own tier, so a misroute cannot be recovered. A single model
answers uncertainty by shrinking toward the mean, which is MSE-optimal; a
cascade is forced to commit. The per-band breakdown confirms the mechanism —
soft routing is worst in the two middle bands (+41%, +48%), exactly where the
tier boundaries sit, and only helps in the top band (−4.4%).

This is consistent with E4, where a binned-distribution head — effectively
soft routing over many tiers with constant experts — also failed to beat a
single regressor.

### The oracle number is the useful part

Routing by the true tier reaches **MSE 40,563 (R² 0.741)**, which is roughly
the top of the public leaderboard. The oracle leaks the answer, so this is not
an achievable result; its value is that it localises the difficulty. Knowing
only whether a house is cheap, mid or expensive removes 46% of the squared
error. Our model captures part of that and the gate recovers 62% of the tier
label, and the gap between 75,218 and 40,563 is almost entirely the gap in
coarse tier discrimination.

This unifies the earlier findings. Tail under-prediction, optimal shrinkage,
and features being the binding constraint are three views of one thing: **the
photograph does not reliably reveal which price tier a house belongs to.**
Anything that improves coarse tier separability is worth far more than
refinements to the regression head or the loss.

---

## Figures

Generated by `src/make_figures.py` into `report/figures/` as vector PDF at
IEEE column widths.

| File | Content |
| --- | --- |
| `fig1_calibration.pdf` | (a) mean signed error by true-price band, (b) mean outcome vs mean prediction by prediction decile — the pair is the argument that tail bias is shrinkage, not a defect |
| `fig2_price_bands.pdf` | MSE per price band, labelled with each band's share of total squared error |
| `fig3_weighting.pdf` | MSE and effective sample size against the weighting exponent α |
| `fig4_representations.pdf` | Out-of-fold MSE by frozen backbone and input resolution |

Units: price is in $1000 USD throughout, so MSE is in ($1000 USD)².

---

# Part II — second round (2026-10-20)

| Reference | MSE | Note |
| --- | --- | --- |
| Predict the training mean | 156,686 | constant-prediction baseline |
| Our 2026-09-23 submission | 100,376 | public LB |
| Our 2026-10-13 submission | **70,097** | public LB, from OOF 74,467 |
| Public leaderboard top | 41,862 | 2026-10-20; 20 teams between 41.9k and 49.2k |

**Out-of-fold to leaderboard calibration.** The 2026-10-13 submission scored
70,097 against an OOF estimate of 74,467, a ratio of 0.941. The September
submission's ratio was 0.993. The difference is refitting: each OOF model
trains on 6,400 rows while the submitted model is refitted on all 8,000. Treat
**LB ~= 0.94 x OOF** as the working conversion.

---

## E7 — DINOv3, and a clean pretraining/architecture decomposition

| Features | MSE | R2 |
| --- | ---: | ---: |
| DINOv3 ViT-H+/16 @336 | **70,818** | 0.548 |
| DINOv3 ViT-H+/16 @256 | 71,344 | 0.545 |
| DINOv3 ViT-H+/16 @224 | 72,396 | 0.538 |
| DINOv3 ViT-L/16 @336 | 74,134 | 0.527 |
| DINOv3 ViT-L/16 @224 | 76,802 | 0.510 |
| DINOv2 ViT-L/14 @336 | 76,807 | 0.510 |
| DINOv3 ConvNeXt-L @288 | 86,454 | 0.448 |
| DINOv3 ConvNeXt-B @288 | 87,744 | 0.440 |
| ConvNeXt-B @288 (IN-22k, supervised) | 98,752 | 0.370 |
| Photographic metadata only (19 features) | 140,538 | 0.103 |

A single DINOv3 ViT-H+ beats the whole four-backbone DINOv2 concatenation with
seed averaging from Part I (74,467). DINOv3 ViT-L @224 (76,802) also matches
DINOv2 ViT-L @336 (76,807): better pretraining data buys the same as a 1.5x
resolution increase, far more cheaply.

**Three factors, separated.** Holding everything else fixed:

| Comparison | Effect |
| --- | ---: |
| Pretraining objective (supervised IN-22k -> self-supervised DINOv3, ConvNeXt-B) | **-11.1%** |
| Architecture (ConvNeXt-L -> ViT-L, both DINOv3) | **-11.2%** |
| Capacity (ConvNeXt-B -> ConvNeXt-L, both DINOv3) | -1.5% |

Doubling parameters is worth almost nothing; what the network was trained to do
and what shape it is are each worth an order of magnitude more. The reading for
the objective: every image here is the same object class, and the invariances
that make a classifier good at telling a house from a car are exactly the ones
that discard within-class appearance, which is where the price signal lives.

Photographic metadata (original dimensions, file size, sharpness, saturation,
sky and vegetation fractions, edge density) reaches R2 0.103 alone and adds
about 0.85% inside a concatenation. Part of it is information the backbone
structurally cannot see, because every image is resized to a square first.

---

## E8 — local tiles, and the planned 2x2 study

Each image split into a 2x2 grid, each tile resized to the full input size,
the four descriptors averaged:

| Features | MSE |
| --- | ---: |
| whole image, DINOv3 ViT-L @224 | 76,802 |
| 2x2 tiles only | 76,800 |
| **whole + tiles** | **74,936** |

Tiles alone and the whole image alone are indistinguishable, but together they
are 2.4% better than either. The local view is complementary, not redundant.
This is the project plan's whole-plus-local-patch contrast, and it holds.

The planned 2x2 study (input type x training objective), on DINOv3 ViT-H+:

| | plain MSE | MSE + ordinal (lambda=0.3) | objective effect |
| --- | ---: | ---: | ---: |
| whole only | 70,818 | 70,722 | -96 |
| whole + local tiles | **70,236** | 70,367 | +131 |
| input effect | -582 | -355 | |

The input-side effect is consistent in sign and 2-4x the seed noise, so it is
real but small. The objective-side effect changes sign between rows, so it is
noise. They do not combine: adding the ordinal task on top of local tiles is
slightly worse than tiles alone. The plan hypothesised complementarity; there
is none.

---

## E9 — resolution is set by the source image, not by the backbone

Each backbone was also run at its own pretraining resolution, to test whether
the earlier sweep had simply been using them out of distribution.

| Backbone | native | @224 | @256 | @336 | @518 |
| --- | ---: | ---: | ---: | ---: | ---: |
| DINOv2 ViT-L/14 | 518 | 78,078 | — | **76,807** | 77,231 |
| DINOv2 ViT-B/14 | 518 | 84,872 | — | **82,414** | 82,401 |
| DINOv3 ViT-L/16 | 256 | 76,802 | 76,112 | **74,134** | — |
| DINOv3 ViT-H+/16 | 256 | 72,396 | 71,344 | **70,818** | — |

**The hypothesis is refuted.** Two families whose pretraining resolutions
differ by a factor of two both optimise at about 336. DINOv2 is worse at its
own native 518 than at 336; DINOv3 is better above its native 256. The common
factor is the source image at 400x300, so about 336 is where there is nothing
left to resolve and further upsampling only interpolates. A cross-family
controlled result that the binding constraint is the data, not the model
configuration.

---

## E11 — regressor comparison (the plan's Ridge vs XGBoost)

On DINOv3 ViT-H+ @336:

| Head | MSE | R2 | wall time |
| --- | ---: | ---: | ---: |
| MLP | **70,818** | 0.548 | 42 s |
| Ridge | 72,624 | 0.537 | 338 s |
| XGBoost | 74,140 | 0.527 | 1,264 s |
| LightGBM | 74,584 | 0.524 | 116 s |

Ridge beats XGBoost, and the MLP beats both while being 30x cheaper than
XGBoost. Ridge outranking the boosted trees reproduces what E4 found on DINOv2
ViT-B, so it is a property of the input rather than of one backbone: no single
dimension of a dense ViT embedding is individually meaningful, which is what
axis-aligned splits need.

The MLP's margin over Ridge also narrows as features improve — 4.3% on DINOv2
ViT-B @224, 2.5% on DINOv3 ViT-H+ @336. The stronger the representation, the
more nearly linear the remaining problem.

---

## E12 — ordinal auxiliary task (the plan's supervision contrast)

Five quantile buckets give four thresholds, trained as auxiliary binary "is
this house worth more than threshold k?" heads beside an untouched scalar
regression output.

| lambda | MSE | vs plain MSE |
| --- | ---: | ---: |
| 0.1 | 70,747 | -0.10% |
| 0.3 | **70,722** | -0.14% |
| 1.0 | 71,132 | +0.44% |
| 0 (plain MSE) | 70,818 | — |

**Neutral.** The best margin is 96 MSE against a seed standard deviation of
130-255, so it is not a result; lambda=1.0 is clearly harmful.

Worth contrasting with the binned-distribution head of E4, which was decisively
worse. The prediction was that the ordinal formulation would avoid that failure
mode, because it *adds* four outputs beside a scalar head still trained on the
metric rather than *replacing* the scalar with 64 competing outputs. That
prediction held — it does not degrade. It also adds nothing, which suggests the
ordering information it injects is already implicit in the regression loss.

---

## E13/E14 — prediction-level fusion

The plan listed prediction-level fusion alongside feature concatenation. Non-
negative least squares over member predictions, with both the mixing weights
and a final linear recalibration fitted out-of-fold (for fold k, on the other
four folds).

A first pass over all 56 stored out-of-fold vectors — excluding the oracle-
routed predictions from E6, which encode the true tier — gave 65,347 raw and
64,909 after linear recalibration. Isotonic recalibration was worse (65,254):
the miscalibration is a uniform tilt (slope 1.073, from averaging several
individually shrunk predictors), which two parameters fix exactly, while
isotonic spends its extra freedom on noise.

The submitted ensemble refits nine members so that test predictions exist:

| MSE | R2 | slope |
| ---: | ---: | ---: |
| **65,278** | 0.583 | 0.998 |

against a best single member of 69,175 — a **5.63%** gain, the largest single
improvement of the second round.

### Ensemble weight does not track member accuracy

| Weight | Member | Own MSE |
| ---: | --- | ---: |
| 0.193 | DINOv2 ViT-L/14 @518 | 77,231 |
| 0.169 | DINOv3 ViT-H+ @256 | 71,344 |
| 0.163 | DINOv3 ViT-H+ @224 + @336 | 70,542 |
| 0.141 | H+@336 + L@336 + photo meta | 69,688 |
| 0.129 | DINOv3 ViT-L 2x2 tiles | 76,800 |
| 0.076 | DINOv3 ViT-L @336 | 74,134 |
| 0.068 | H+@336 + L@336 + tiles + meta | 69,832 |
| 0.037 | DINOv3 ViT-H+ @336 | 70,818 |
| 0.033 | 6-way concatenation | **69,175** (best single) |

The best single member receives the smallest weight. The two largest weights go
to the only member from a different pretraining family (DINOv2) and the only
member with a different input geometry (local tiles); the other seven are
variants of the same DINOv3 global representation and are too correlated to
each earn much. The 56-member blend showed the same pattern: the tiered routing
predictions, which lost badly alone (+14.5% and +26.3%), still drew weights of
0.081 and 0.027.

**On this task, representation diversity is worth more than member accuracy.**
It also rehabilitates E6: tiered routing fails as a predictor and still
contributes as an ensemble member, because it makes different mistakes.

---

## What was ruled out as an explanation for the leaderboard gap

Our 70,097 sits far behind the 41.9k-49.2k band occupied by twenty teams, so
each cheap systematic explanation was tested and rejected:

| Hypothesis | Verdict |
| --- | --- |
| Image ids leak price | Rejected: Pearson -0.006; id-adjacent pairs differ by 417 vs 429 at random |
| Train/test near-duplicates | Rejected: 0.37% of test images at pHash Hamming <= 6 |
| Tract housing exploitable by retrieval | Rejected: k-NN on DINOv3 features peaks at 98,839 (k=10) against 70,818 for an MLP on the same features; k=1 gives R2 0.035 |
| Backbones used below native resolution | Rejected: see E9 |

What remains is modelling capacity — end-to-end fine-tuning and large ensembles
— which the compute budget did not allow.

---

## Limitations

**End-to-end fine-tuning was not adequately evaluated.** Two attempts were
made. The first (DINOv2 ViT-B @280, 20 epochs) returned 73,481 on fold 0,
behind the frozen-feature concatenation's 69,827 on the same fold — but its
console output was lost to buffering, so under-training could not be
distinguished from a failed run. The second (DINOv3 ViT-L @256, 40 epochs,
mixup, layer-wise LR decay) was stopped after 135 minutes: one fold costs about
two hours on the available 8 GB GPU and five folds would cost over ten.

This is a gap, not a negative result, and should be read as such. The frozen-
feature route was chosen deliberately: cached features make an experiment loop
that runs in seconds, which is what allowed seventy-plus controlled
configurations to be compared, where the same compute would have bought two or
three fine-tuning runs. Given the oracle result in E6, which shows large
headroom in coarse tier separability, systematic fine-tuning and multi-model
ensembling are the obvious first items of future work.

---

## Final configuration

Nine frozen-feature MLPs blended by non-negative least squares with linear
recalibration. Members span two pretraining families (DINOv2, DINOv3), four
input resolutions (224/256/336/518), two input geometries (whole frame, 2x2
tiles) and photographic metadata.

| | OOF MSE | R2 |
| --- | ---: | ---: |
| Constant-prediction baseline | 156,686 | 0 |
| Best single frozen backbone (DINOv3 ViT-H+ @336) | 70,818 | 0.548 |
| Best feature concatenation (6-way, 3 seeds) | 68,522 | 0.563 |
| **Nine-member blend** | **65,278** | **0.583** |

Predicted public leaderboard about 61,400 at the measured 0.94 conversion,
against 70,097 for the submission currently on the board.
