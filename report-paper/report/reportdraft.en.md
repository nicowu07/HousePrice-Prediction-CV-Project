# Evaluating Global and Local Visual Features for House Price Prediction

## Introduction

A very important step in putting a property on the market is setting a reasonable price, and real estate agents usually combine property characteristics, market conditions and information on comparable transactions to recommend a listing price [1]. A reasonable valuation can provide an agency with a good starting point for negotiation, and can also help real estate platforms determine which properties need further price review. Although these potential uses reflect the practical value of house price prediction, estimating a price from a single image is still difficult. A photograph can contain information on architectural style, the visible extent of the building and the surrounding environment, but does not directly provide the floor area, precise location or interior condition. At the same time, different viewing angles, lighting and occlusion of the same house can also affect the estimation of its price.

In this project, we used 8,000 images with price labels to train lightweight regressors on frozen pretrained visual features, to accommodate limited labels and compute. During model training, we focused on studying whether combining whole-image and local features could reduce prediction error and whether price-order auxiliary supervision could further improve prediction. Finally, we further combined the relevant experimental models to investigate whether using different models together would increase prediction accuracy.



## Method

Our overall idea is to first convert images into numerical features and then learn the relationship between these features and house prices. We first resized all images to the specified square input size, then used the DINOv3 [2] image model with frozen parameters to extract features, and trained a small regression network using price labels. To test whether local views could provide additional information, we separately processed the full image and the four non-overlapping local tiles obtained by dividing the original image evenly into a 2×2 grid. Each image tile was resized and its features were extracted separately, and the four groups of features were then averaged and concatenated with the whole-image features before being fed into the regression network. We then used controlled experiments to evaluate whether this whole-image + local combination was better than using only the whole image.

For price prediction, we used a small neural network (MLP). The input features were first standardised using statistics from the training portion; the target prices were also first log-transformed and standardised to put them on a more concentrated numerical scale, and the predictions were then transformed back into prices. We also studied an auxiliary task that asked the network to judge whether the house price was greater than four price thresholds at the same time (these thresholds were determined from the price distribution in the corresponding training portion, dividing the houses into five groups with approximately equal numbers). Therefore, while the network was learning specific numerical values, it was also learning the ordering of prices from low to high. The training objective consists of the price prediction error and the error in judging price order, with the parameter λ controlling the weight of the auxiliary task. The two tasks share the regression network, but the final valuation only uses the price prediction output. All results are evaluated on the original price scale after the inverse transformation.

$$
\mathcal{L}=\mathcal{L}_{\mathrm{price}}+\lambda\mathcal{L}_{\mathrm{order}}.
$$

**Table 1　Training objective and notation**

| Symbol | Meaning | Calculation or value used in this project |
| --- | --- | --- |
| $\mathcal{L}$ | Total training loss | Sum of the price prediction loss and the weighted auxiliary loss |
| $\mathcal{L}_{\mathrm{price}}$ | Price prediction loss | Mean squared error of standardised log(price) |
| $\mathcal{L}_{\mathrm{order}}$ | Auxiliary loss for the four price-threshold judgements | Binary cross-entropy for the four binary judgements, averaged over samples and thresholds |
| $\lambda$ | Weight of the auxiliary loss | 0.3 in the combined experiments; 0 for regression only |

After the above price prediction, we further calibrated it, using a correction curve to adjust the predicted values to bring them closer to the actual prices, and then applied the curve to the validation samples. Apart from adjusting the output of an individual predictor, we also studied whether different predictors were complementary, namely how different visual features, input sizes and local views would produce different prediction errors. To further study the effect of combining different image information, we added frozen DINOv2 [3] features alongside DINOv3, and examined combinations of whole-image features at different input sizes, local features, and image statistics such as image dimensions and sharpness. Each feature configuration was used to train a separate MLP price predictor, with the specific compositions shown in Table 3. We combined the outputs of nine existing regression predictors using weights, and then applied an overall linear correction to the combined result. We divided the data into five parts and took turns using one part to evaluate the ensemble. Each time, we used the existing predicted prices and true prices from the other four parts to determine the ensemble weights and overall correction parameters, and then applied these parameters to the currently held-out part. The specific regression network structure, training parameters and post-processing settings are shown in Table 2.



**Table 2　Regression and post-processing configuration**

| Setting | Configuration used in this project |
| --- | --- |
| Network structure | Two hidden layers with widths of 512 and 256; each layer uses BatchNorm, GELU and dropout (0.3) |
| Output | 1 scalar for regression only; adding ordinal supervision introduces 4 additional threshold judgements, while the final valuation still uses only the scalar output |
| Optimiser | AdamW; weight decay 0.01 |
| Learning rate schedule | OneCycle; maximum learning rate 0.001, with the first 15% of scheduling steps used to increase the learning rate |
| Training budget | Each regression head is trained for 40 epochs, with 256 samples per batch |
| Auxiliary thresholds | The 20%, 40%, 60% and 80% price quantiles of the corresponding training portion |
| Individual predictor calibration | Within each outer training portion, ordinary random five-fold splits generate held-out predictions; these predictions and the training labels are used to fit an isotonic correction curve, which is then applied to the outer validation predictions |
| Ensemble and output range | The nine predictors use non-negative least-squares weights, followed by linear correction; final predictions are limited to 195–2,000 thousand dollars |

**Table 3　Inputs to the nine ensemble members**

| Predictor | Input feature groups | Number of feature groups |
| --- | --- | ---: |
| P1 | H+@336 + H+@256 + L@336 + D2L@336 + T + M | 6 |
| P2 | H+@336 + L@336 + M | 3 |
| P3 | H+@336 + L@336 + T + M | 4 |
| P4 | H+@224 + H+@336 | 2 |
| P5 | H+@336 | 1 |
| P6 | H+@256 | 1 |
| P7 | L@336 | 1 |
| P8 | T | 1 |
| P9 | D2L@518 | 1 |

*Note: H+ and L denote DINOv3 ViT-H+ and ViT-L, respectively, and D2L denotes DINOv2 ViT-L; @s denotes an s×s-pixel input. T is the mean of the features from the 2×2 local tiles extracted using L@224; M consists of 19 image statistics (such as dimensions and sharpness).*



## Experiments and Results

To examine the prediction performance of different designs, this experiment uses a controlled-variable approach to compare price prediction methods, image inputs, the auxiliary task and prediction combinations under common evaluation conditions. The dataset was assigned to a fixed five-group split, keeping the price distributions in the parts as similar as possible. Predictions are evaluated using mean squared error (MSE), where lower values are better. Under this common setting, we tested the accuracy of three price prediction methods: MLP, Ridge and XGBoost [4]. All three used the same DINOv3 ViT-H+ image features extracted using 336×336-pixel inputs. Table 4(a) shows that the MSEs of MLP, Ridge and XGBoost were 70,818, 72,624 and 74,140, respectively, with MLP having the lowest error in the tested settings. Therefore, the subsequent main comparisons all used MLP, allowing us to examine the effects of image input and the auxiliary task separately while keeping the price prediction method unchanged.

**Table 4(a)　Regressors with identical frozen features**

| Regressor | OOF MSE ↓ |
| --- | ---: |
| MLP | **70,818** |
| Ridge | 72,624 |
| XGBoost | 74,140 |

*Note: All three use whole-image features extracted by DINOv3 ViT-H+ from 336×336-pixel inputs and the same fixed grouped five-fold split, with detected near-duplicate images kept in the same fold. The following result tables all report OOF MSE on the original price scale, in (thousand dollars)², rounded to integers.*

After fixing the price prediction method, we next checked whether using full photographs, local images or a combination of the two would affect the results. To keep the image model consistent in this view comparison, both whole-image and local features were extracted using DINOv3 ViT-L with 224*224-pixel inputs. The MSEs of using only the whole image and only local features were almost the same, at 76,802 and 76,800, respectively; concatenating the two groups of features reduced the MSE to 74,936, approximately 2.4% lower than using the whole image (Table 4(b)). Therefore, in the current configuration, the value of local features mainly lies in supplementing whole-image information rather than replacing it on their own.

**Table 4(b)　Whole-image and local views with the same encoder**

| Input features | OOF MSE ↓ |
| --- | ---: |
| Whole image | 76,802 |
| Mean features from 2×2 local tiles | 76,800 |
| Concatenated whole-image and local features | **74,936** |

*Note: Both the whole image and each local tile use DINOv3 ViT-L with 224×224-pixel inputs; all regressors are regression-only MLPs. The three rows use the same evaluation split, and concatenation reduces MSE by approximately 2.4% relative to the whole image. These are single-seed results.*

Besides testing different combinations of views, we also examined whether the auxiliary task of judging higher or lower prices (ordinal) could improve prediction. In this experiment, we used the whole-image configuration with ViT-H+ and 336×336-pixel inputs, which had a lower error, and separately examined whether to add ViT-L@224 local features and whether to enable the auxiliary task (Table 5). The results show that the auxiliary task itself only slightly reduced the error of the whole-image configuration, while slightly increasing it when whole-image and local features were combined. Because the changes were small and inconsistent in direction, we cannot determine whether the price-order auxiliary task can provide a stable additional benefit.

**Table 5　Local features and ordinal supervision**

| Input configuration | Continuous regression only (λ=0) | Continuous regression + ordinal (λ=0.3) |
| --- | ---: | ---: |
| Global DINOv3 ViT-H+@336 | 70,818 | 70,722 |
| Global DINOv3 ViT-H+@336 + local DINOv3 ViT-L@224 | **70,236** | 70,367 |

*Note: Values are OOF MSE in (thousand dollars)². Local features are the mean features of the four tiles in a 2×2 grid. The global and local branches use different encoders and input sizes, so this comparison concerns the complete configuration with this local branch added; the view comparison using the same encoder is shown in Table 4(b). All four settings have single-seed results, and the small differences have not undergone complete repeated validation.*

Finally, we also tried some combinations of different strategies (Table 6), concatenating six groups of features and feeding them into the same MLP, which gave an MSE of 69,175; training the same configuration with three different random initialisations and then averaging the predictions reduced the MSE to 68,522. Then, combining the price outputs of nine differently configured predictors using weights gave an MSE of 65,278, approximately 5.6% lower than the best member trained in a single run (Table 6). Because member selection reused the development evaluation results and the overall process has not undergone fully independent validation, we cannot determine whether the combination of nine predictors is the best combination; it simply had the lowest MSE among all the combinations we tried.

**Table 6　Feature combination and prediction averaging**

| Configuration | What is combined and how it is trained | OOF MSE ↓ |
| --- | --- | ---: |
| Six concatenated feature groups + MLP | P1 in Table 3; one random initialisation | 69,175 |
| Same six-group configuration, prediction averaging over three seeds | P1 trained with three different random initialisations, followed by averaging the price predictions | 68,522 |
| Nine-member prediction ensemble | P1–P9 in Table 3; non-negative least-squares weights + linear correction | **65,278** |

*Note: MSE is in (thousand dollars)². The approximately 5.6% reduction uses 69,175 as the comparison baseline. The nine-member ensemble uses the single-seed predictions of each member, rather than directly adding members to the three-seed average; the three rows also do not form a complete ablation that changes only one factor. The value of 65,278 is a local development OOF estimate; member selection and the overall process have not been fully independently validated, and it is not a measured Kaggle score.*



## Error Analysis and Discussion

Although the prediction error was further reduced through successive combinations, this does not mean that this improvement is effective for images of houses in all price bands. Analysis of the predictions from the system using the combination strategy shows that expensive houses have a large influence on the overall MSE: the 1,279 images with true prices of at least 1,000 thousand dollars account for 16.0% of the samples but contribute 55.7% of the total squared error; there is also a clear direction of bias: the mean “prediction minus true price” bias in the 1,500–2,000 thousand dollar band is −532.1 thousand dollars, compared with +8.6 thousand dollars in the 700–1,000 band (Figure 1).

![Figure 1: Error distribution, bias by price band and calibration of the final nine-member ensemble](</Users/lhzzzzz/Desktop/COMP90086 CV/final_project/report/images/figure1_ensemble_error_analysis.png>)

**Figure 1　Error and calibration diagnostics for the final ensemble.** (a) Comparison of the share of samples and the share of total squared error by true-price band; (b) mean “prediction−true price” bias in each true-price band, where n is the number of samples in that band; (c) predictions sorted from low to high and divided into ten equally sized groups of 800 images, comparing the mean prediction and mean true price in each group, with the dashed line indicating equality. Prices and biases are in thousand dollars, and the final price band includes 2,000. All three panels use the same final ensemble OOF predictions for 8,000 images; bin means close to the diagonal do not imply accurate estimates for individual images or remove the limitations of development evaluation.

To further illustrate these MSEs, we show four specific cases from the final model (Figure 2). The facade in 4244.jpg is clear, and its predicted price of 768.98 thousand dollars is close to its true price of 769 thousand dollars, so it can serve as a relatively accurate reference. The price of 2138.jpg is underestimated by 1468 thousand dollars, possibly because its wooden exterior walls and the surrounding land with little greenery do not sufficiently reflect its higher true price. The price of 5350.jpg is overestimated by 850 thousand dollars, possibly because visual features such as its tall arched windows, warm lighting and lawn are associated with higher-value houses. In 2409.jpg, there is considerable occlusion from trees in the foreground, which affected the model's judgement of the building's price.

![Figure 2: Final ensemble OOF prediction cases for 4244, 2138, 5350 and 2409](</Users/lhzzzzz/Desktop/COMP90086 CV/final_project/report/images/figure2_ensemble_cases.png>)

**Figure 2　Four illustrative OOF predictions from the final ensemble.** (a) 4244.jpg: a prediction close to the true price; (b) 2138.jpg: a severe underestimate; (c) 5350.jpg: a clear overestimate; (d) 2409.jpg: foreground trees obscure part of the building, and the price is underestimated. Each image is labelled with its image ID, true price, predicted price and “prediction−true price” error, in thousand dollars, with values rounded to two decimal places. The original images retain their full frame and original aspect ratio. These four examples follow the cases specified for the project and all use the final nine-member ensemble's OOF predictions; they illustrate individual situations rather than the overall distribution, and visible image characteristics cannot establish the causes of errors.

In addition, this study was also limited by available compute and runtime, which is also why we mainly used frozen visual features, so if there is still an opportunity to further optimise and improve this experiment in the future, we can fine-tune the visual model to test whether it helps house price prediction. There is still room for further optimisation of the current results.



## Conclusion

Taken together, the above comparisons and error analysis show that frozen visual features allow us to effectively estimate house prices even with a limited dataset and compute (with an MSE clearly lower than the baseline MSE of 151384.8 provided on Kaggle). In the tested configurations, combining whole-image and local features performed better than using either view alone, while the price-order auxiliary task did not show a clear additional benefit. Averaging predictions from repeated training runs and combining different predictors further reduced the development error, but expensive houses were still prone to underestimation.





## Reference

[1] National Association of REALTORS®, “Consumer Guide: What Goes Into Pricing Your Home.” [Online]. Available: [Official guide](https://www.nar.realtor/the-facts/consumer-guide-what-goes-into-pricing-your-home). Accessed: Oct. 24, 2026.

[2] O. Siméoni et al., “DINOv3,” arXiv:2508.10104, 2025. [Paper](https://arxiv.org/abs/2508.10104).

[3] M. Oquab et al., “DINOv2: Learning Robust Visual Features without Supervision,” Transactions on Machine Learning Research, 2024. [Paper](https://openreview.net/forum?id=a68SUt6zFt).

[4] T. Chen and C. Guestrin, “XGBoost: A Scalable Tree Boosting System,” in Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2016, pp. 785–794. [Paper](https://arxiv.org/abs/1603.02754).
