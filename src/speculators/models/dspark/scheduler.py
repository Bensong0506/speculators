"""Hardware-aware prefix scheduler for DSpark (paper Algorithm 1).

Given, for each active request, the per-position confidence scores
``c_{r,1..gamma}`` (calibrated conditional acceptance probabilities from the
confidence head — see :mod:`speculators.models.dspark.calibration`) and a
profiled engine throughput curve ``SPS(B)`` (steps-per-second as a function of
the forward-pass batch size ``B`` in tokens), this picks a per-request
verification length ``l_r in {0..gamma}`` that maximizes expected system-wide
token throughput ``Theta = tau * SPS(B)``.

The module is deliberately pure-Python and engine-agnostic (no torch): it is
the portable core to be called from a serving engine's spec-decode step. The
engine supplies the confidence scores (per request) and the profiled ``SPS``
cost table; this returns the scheduled prefix lengths.

Key facts used (paper Section 3.2.2):
  * Prefix survival ``a_{r,j} = prod_{i<=j} c_{r,i}`` is monotonically
    non-increasing in ``j``, so a global descending sort by ``a_{r,j}``
    automatically respects intra-request order (j-1 admitted before j).
  * Admission is a greedy walk down that sorted list; the early-stop on the
    first throughput drop keeps the decision non-anticipating (lossless).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Callable

__all__ = [
    "make_sps_lookup",
    "schedule_prefix_lengths",
]

SpsFn = Callable[[int], float]


def make_sps_lookup(cost_table: Sequence[float]) -> SpsFn:
    """Build an ``SPS(B)`` callable from a profiled cost table.

    ``cost_table[B]`` is the steps-per-second at forward batch size ``B``.
    Out-of-range ``B`` clamps to the nearest profiled endpoint (batch sizes
    below/above what was profiled reuse the closest measured value).
    """
    if len(cost_table) == 0:
        raise ValueError("cost_table must be non-empty")
    table = [float(v) for v in cost_table]
    max_b = len(table) - 1

    def sps(batch_size: int) -> float:
        if batch_size < 0:
            batch_size = 0
        if batch_size > max_b:
            batch_size = max_b
        return table[batch_size]

    return sps


def schedule_prefix_lengths(
    confidences: Sequence[Sequence[float]],
    sps: SpsFn,
    *,
    gamma: int | None = None,
) -> list[int]:
    """Return per-request verification lengths maximizing ``tau * SPS(B)``.

    Args:
        confidences: ``confidences[r]`` is the per-position confidence sequence
            ``c_{r,1}, ..., c_{r,gamma}`` for request ``r`` (each in ``[0, 1]``).
        sps: profiled throughput ``SPS(B)`` (steps/sec) for forward batch size
            ``B`` in tokens; build one with :func:`make_sps_lookup`.
        gamma: optional cap on the number of draft positions considered per
            request. Defaults to each request's own confidence length.

    Returns:
        ``l_star`` — a list of length ``R`` of scheduled prefix lengths in
        ``{0..gamma}``. ``l_star[r] == 0`` means only the guaranteed bonus token
        is verified for request ``r`` (no speculative tokens admitted).
    """
    num_requests = len(confidences)
    selected = [0] * num_requests
    if num_requests == 0:
        return selected

    # Candidate space E = {(a_{r,j}, r, j) | a_{r,j} > 0}. Because a_{r,j} is
    # monotone non-increasing in j, appending in increasing j and sorting by a
    # descending keeps each request's positions in order.
    candidates: list[tuple[float, int, int]] = []
    for r, conf_seq in enumerate(confidences):
        limit = len(conf_seq) if gamma is None else min(int(gamma), len(conf_seq))
        survival = 1.0
        for j in range(1, limit + 1):
            survival *= float(conf_seq[j - 1])
            if survival > 0.0:
                candidates.append((survival, r, j))
            else:
                # c >= 0 => all further cumulative products are also 0.
                break
    candidates.sort(key=lambda item: item[0], reverse=True)

    # Baseline: every request contributes its bonus token (batch B=R, tau=R).
    batch_size = num_requests
    expected_accepts = float(num_requests)
    best_throughput = expected_accepts * sps(batch_size)

    for survival, request_idx, prefix_len in candidates:
        batch_size += 1
        expected_accepts += survival
        throughput = expected_accepts * sps(batch_size)
        if throughput > best_throughput:
            best_throughput = throughput
            selected[request_idx] = prefix_len
        else:
            # Early stop: unimodal objective + non-anticipating admission.
            break

    return selected
