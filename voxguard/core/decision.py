"""Decision theory and calibration used by every plugin.

Confirm rule (expected cost):
    confirm  iff  P(error) * cost_error  >  cost_confirm

Calibration: the rule is only correct if P(error) is calibrated, so we provide
temperature scaling and expected calibration error (ECE).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CostModel:
    """Relative costs. Only ratios matter."""
    cost_error: float = 10.0     # a wrong value reaching the backend
    cost_confirm: float = 1.0    # one extra confirmation turn

    @property
    def threshold(self) -> float:
        """P(error) above which confirming is optimal."""
        return self.cost_confirm / self.cost_error

    def should_confirm(self, p_error: float) -> bool:
        return p_error * self.cost_error > self.cost_confirm


# Per-field defaults: wrong phone numbers lose the customer, wrong party size is cheap.
DEFAULT_COSTS: dict[str, CostModel] = {
    "phone": CostModel(cost_error=20.0, cost_confirm=1.0),
    "card": CostModel(cost_error=50.0, cost_confirm=1.0),
    "order_id": CostModel(cost_error=15.0, cost_confirm=1.0),
    "name": CostModel(cost_error=8.0, cost_confirm=1.0),
    "email": CostModel(cost_error=15.0, cost_confirm=1.0),
    "number": CostModel(cost_error=5.0, cost_confirm=1.0),
}


def combine_agreement(p_model_wrong: float, p_shadow_wrong: float, agree: bool) -> float:
    """P(final value wrong) from two recognizers assumed conditionally independent.

    If both agree, the value is wrong only if both are wrong in the same way;
    we upper-bound that by p_m * p_s. If they disagree, at least one is wrong and
    the probability the model's value is the wrong one is p_m / (p_m + p_s)
    under a symmetric-error approximation.
    """
    p_m = float(np.clip(p_model_wrong, 1e-6, 1 - 1e-6))
    p_s = float(np.clip(p_shadow_wrong, 1e-6, 1 - 1e-6))
    if agree:
        return p_m * p_s
    return p_m * (1 - p_s) / (p_m * (1 - p_s) + p_s * (1 - p_m))


class TemperatureScaler:
    """Fit a single temperature T so that sigmoid(logit / T) is calibrated."""

    def __init__(self) -> None:
        self.T = 1.0

    @staticmethod
    def _nll(logits: np.ndarray, labels: np.ndarray, T: float) -> float:
        p = 1 / (1 + np.exp(-logits / T))
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))

    def fit(self, confidences: np.ndarray, correct: np.ndarray) -> TemperatureScaler:
        c = np.clip(np.asarray(confidences, float), 1e-6, 1 - 1e-6)
        logits = np.log(c / (1 - c))
        y = np.asarray(correct, float)
        grid = np.exp(np.linspace(np.log(0.1), np.log(10), 400))
        self.T = float(min(grid, key=lambda t: self._nll(logits, y, t)))
        return self

    def transform(self, confidences: np.ndarray | float) -> np.ndarray | float:
        c = np.clip(np.asarray(confidences, float), 1e-6, 1 - 1e-6)
        out = 1 / (1 + np.exp(-np.log(c / (1 - c)) / self.T))
        return float(out) if np.ndim(out) == 0 else out


def expected_calibration_error(confidences, correct, n_bins: int = 10) -> float:
    c = np.asarray(confidences, float)
    y = np.asarray(correct, float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (c > lo) & (c <= hi)
        if m.any():
            ece += m.mean() * abs(c[m].mean() - y[m].mean())
    return float(ece)
