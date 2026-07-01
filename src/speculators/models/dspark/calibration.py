"""Sequential Temperature Scaling (STS) for the DSpark confidence head.

The hardware-aware scheduler (:mod:`speculators.models.dspark.scheduler`) needs
the *absolute* magnitude of the cumulative prefix-survival probability
``a_k = prod_{i<=k} c_i`` to estimate throughput, but raw neural confidence
scores are typically over-confident. STS (paper Section 3.2.1) calibrates the
cumulative product left-to-right: at each position ``k`` it grid-searches a
single temperature ``T_k`` that minimizes the Expected Calibration Error (ECE)
of the cumulative product, holding the already-fitted temperatures of earlier
positions fixed. Temperature scaling is order-preserving, so it fixes the
magnitudes without disturbing the relative ranking the head learned.

Pure-Python and engine-agnostic: fit on validation ``(confidence, prefix-label)``
arrays dumped from an eval run, then apply the temperatures at serving time.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

__all__ = [
    "temperature_scale",
    "expected_calibration_error",
    "calibrated_prefix_survival",
    "fit_sequential_temperature_scaling",
    "default_temperature_grid",
]

_EPS = 1e-8


def _logit(prob: float) -> float:
    prob = min(max(prob, _EPS), 1.0 - _EPS)
    return math.log(prob / (1.0 - prob))


def temperature_scale(prob: float, temperature: float) -> float:
    """Order-preserving temperature scaling of a probability in (0, 1)."""
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    return 1.0 / (1.0 + math.exp(-_logit(prob) / temperature))


def default_temperature_grid(
    *, low: float = 0.25, high: float = 8.0, num: int = 64
) -> list[float]:
    """A log-spaced 1D grid of temperatures for the STS grid search."""
    if num < 2:
        return [1.0]
    log_low, log_high = math.log(low), math.log(high)
    step = (log_high - log_low) / (num - 1)
    return [math.exp(log_low + i * step) for i in range(num)]


def expected_calibration_error(
    preds: Sequence[float],
    labels: Sequence[float],
    *,
    num_bins: int = 15,
) -> float:
    """Binned ECE between predicted probabilities and binary labels.

    ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|, over ``num_bins`` equal-width
    probability bins. Returns ``nan`` when there are no samples.
    """
    total = len(preds)
    if total == 0:
        return float("nan")
    bin_weight = [0.0] * num_bins
    bin_pred = [0.0] * num_bins
    bin_label = [0.0] * num_bins
    for pred, label in zip(preds, labels):
        idx = int(pred * num_bins)
        if idx >= num_bins:
            idx = num_bins - 1
        elif idx < 0:
            idx = 0
        bin_weight[idx] += 1.0
        bin_pred[idx] += pred
        bin_label[idx] += label
    ece = 0.0
    for b in range(num_bins):
        weight = bin_weight[b]
        if weight <= 0.0:
            continue
        avg_pred = bin_pred[b] / weight
        avg_label = bin_label[b] / weight
        ece += (weight / total) * abs(avg_pred - avg_label)
    return ece


def calibrated_prefix_survival(
    confidence: Sequence[float],
    temperatures: Sequence[float],
) -> list[float]:
    """Cumulative product of temperature-scaled per-position confidences.

    ``out[k] = prod_{i<=k} temperature_scale(confidence[i], temperatures[i])`` —
    the calibrated prefix-survival probabilities used by the scheduler.
    """
    out: list[float] = []
    running = 1.0
    for i, conf in enumerate(confidence):
        temp = temperatures[i] if i < len(temperatures) else 1.0
        running *= temperature_scale(conf, temp)
        out.append(running)
    return out


def fit_sequential_temperature_scaling(
    confidences: Sequence[Sequence[float]],
    prefix_labels: Sequence[Sequence[float]],
    *,
    block_size: int,
    temperature_grid: Sequence[float] | None = None,
    num_bins: int = 15,
) -> list[float]:
    """Fit one temperature per draft position (paper Section 3.2.1).

    Args:
        confidences: per-sample per-position raw confidence ``c_i in (0, 1)``.
            Samples may be shorter than ``block_size`` (a shorter draft); each
            position is fit only over the samples that reached it.
        prefix_labels: per-sample per-position prefix-survival label (``1`` if
            every draft token up to and including position ``i`` was accepted,
            else ``0``), aligned with ``confidences``.
        block_size: number of draft positions (gamma).
        temperature_grid: candidate temperatures; defaults to
            :func:`default_temperature_grid`.
        num_bins: ECE histogram resolution.

    Returns:
        ``temperatures`` — a list of length ``block_size``; positions with no
        validation coverage keep temperature ``1.0`` (identity).
    """
    grid = list(temperature_grid) if temperature_grid else default_temperature_grid()
    temperatures = [1.0] * block_size

    # cumprod_prev[s] = prod_{i<k} calibrated(confidences[s][i], temperatures[i]).
    cumprod_prev = [1.0] * len(confidences)

    for k in range(block_size):
        # Collect samples that reached position k (both confidence + label).
        active = [
            s
            for s in range(len(confidences))
            if k < len(confidences[s]) and k < len(prefix_labels[s])
        ]
        if not active:
            # No coverage at this position: keep identity, advance cumprod.
            for s in range(len(confidences)):
                if k < len(confidences[s]):
                    cumprod_prev[s] *= temperature_scale(confidences[s][k], 1.0)
            continue

        best_temp = 1.0
        best_ece = float("inf")
        for temp in grid:
            preds = [
                cumprod_prev[s] * temperature_scale(confidences[s][k], temp)
                for s in active
            ]
            labels = [float(prefix_labels[s][k]) for s in active]
            ece = expected_calibration_error(preds, labels, num_bins=num_bins)
            if ece < best_ece:
                best_ece = ece
                best_temp = temp
        temperatures[k] = best_temp

        # Lock in T_k and advance the running cumulative product.
        for s in range(len(confidences)):
            if k < len(confidences[s]):
                cumprod_prev[s] *= temperature_scale(confidences[s][k], best_temp)

    return temperatures
