"""Price discretisation for the distribution head.

Instead of regressing a scalar, the model predicts a distribution over price
bins and we decode its expectation. The motivation is that the competition
metric is MSE on raw price, whose Bayes-optimal predictor is E[price|x], and
the expectation of a predicted distribution estimates exactly that -- without
the log-to-raw bias correction a log-scale point regression needs.

Three details matter and are implemented here:

*   **Log-spaced edges.** Prices span 195-2000 with most mass low down. Linear
    bins would give the cheap half almost no resolution. Equal width in log
    space matches where the data actually is.

*   **Bin centres are the mean training price inside the bin**, not the
    geometric centre of the interval. Within a bin the price distribution is
    still skewed, so the interval midpoint is a biased stand-in for the
    conditional mean of that bin, and the bias would survive decoding.

*   **Soft targets.** A one-hot target asserts that the bin boundary is
    meaningful, which it is not -- a house at 999 and one at 1001 are the same
    house. Spreading the target over neighbouring bins with a Gaussian kernel
    in log space encodes the ordinal structure of price, which plain
    classification throws away. This is what the ordinal auxiliary loss in our
    project plan was reaching for; label-distribution learning achieves it
    directly in the main loss rather than as a side task.
"""

from __future__ import annotations

import numpy as np

PRICE_MIN = 195.0
PRICE_MAX = 2000.0


class PriceBinner:
    """Log-spaced price bins with soft targets and expectation decoding."""

    def __init__(self, n_bins: int = 64, smoothing: float = 1.5) -> None:
        self.n_bins = n_bins
        self.smoothing = smoothing  # kernel width, in units of bin width
        self.edges: np.ndarray | None = None
        self.centers: np.ndarray | None = None
        self.log_centers: np.ndarray | None = None

    def fit(self, prices: np.ndarray) -> "PriceBinner":
        """Fit edges and centres. Call on the training portion only."""
        lo, hi = np.log(PRICE_MIN), np.log(PRICE_MAX)
        self.edges = np.exp(np.linspace(lo, hi, self.n_bins + 1))
        # Nudge the outer edges so no training price falls outside.
        self.edges[0], self.edges[-1] = PRICE_MIN - 1e-6, PRICE_MAX + 1e-6

        idx = np.clip(np.digitize(prices, self.edges) - 1, 0, self.n_bins - 1)
        centers = np.empty(self.n_bins)
        for k in range(self.n_bins):
            mask = idx == k
            if mask.any():
                centers[k] = prices[mask].mean()
            else:
                # Empty bin: fall back to the geometric midpoint of the interval.
                centers[k] = float(np.sqrt(self.edges[k] * self.edges[k + 1]))
        self.centers = centers
        self.log_centers = np.log(centers)
        return self

    def soft_targets(self, prices: np.ndarray) -> np.ndarray:
        """(n, n_bins) target distributions, Gaussian in log price."""
        if self.log_centers is None:
            raise RuntimeError("call fit() first")
        bin_width = (np.log(PRICE_MAX) - np.log(PRICE_MIN)) / self.n_bins
        sigma = self.smoothing * bin_width
        d = np.log(prices)[:, None] - self.log_centers[None, :]
        logits = -0.5 * (d / sigma) ** 2
        logits -= logits.max(axis=1, keepdims=True)
        w = np.exp(logits)
        return (w / w.sum(axis=1, keepdims=True)).astype(np.float32)

    def decode_mean(self, probs: np.ndarray) -> np.ndarray:
        """E[price] under the predicted distribution."""
        return probs @ self.centers

    def decode_std(self, probs: np.ndarray) -> np.ndarray:
        """Predictive standard deviation -- the model's own uncertainty.

        Used in error analysis to test whether the model knows when it does
        not know, which a point regressor cannot tell us.
        """
        mean = self.decode_mean(probs)
        var = probs @ (self.centers ** 2) - mean ** 2
        return np.sqrt(np.clip(var, 0, None))
