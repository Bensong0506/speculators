"""Metrics and loss functions for DSpark draft model training."""

from typing import Any

import torch
import torch.nn.functional as F

from speculators.models.metrics import (
    compute_accuracy_multi_step,
    dflash_loss_decay,
)

_EPS = 1e-6


def _weighted_mean(
    values: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    numerator = (values * weights).sum()
    denominator = weights.sum()
    return numerator / (denominator + _EPS), numerator, denominator


def compute_dspark_metrics(
    logits: torch.Tensor,  # shape: [1, num_anchors*block_size, draft_vocab_size]
    targets: torch.Tensor,  # shape: [1, num_anchors*block_size, draft_vocab_size]
    confidence_logits: torch.Tensor | None,  # shape: [1, num_anchors*block_size]
    loss_mask: torch.Tensor,  # shape: [1, num_anchors*block_size]
    block_size: int,
    *,
    ce_loss_alpha: float = 0.1,
    l1_loss_alpha: float = 0.9,
    confidence_head_alpha: float = 1.0,
    gamma: float = 4.0,
    ce_label_ids: torch.Tensor | None = None,  # shape: [1, num_anchors*block_size]
    ce_label_valid: torch.Tensor | None = None,  # shape: [1, num_anchors*block_size]
) -> tuple[torch.Tensor, dict]:
    """Compute DSpark's CE + L1 + confidence training objective.

    The confidence target is the rejection-sampling acceptance proxy
    ``1 - 0.5 * ||softmax(logits) - softmax(targets)||_1``.

    ``ce_label_ids`` selects the cross-entropy label source. When provided it is
    the paper-faithful ground-truth realized token (already mapped into the draft
    vocabulary), and ``ce_label_valid`` masks out positions whose ground-truth
    token falls outside the reduced draft vocabulary (out-of-vocab). When
    ``ce_label_ids`` is ``None`` the CE label falls back to the target model's
    top-1 (``argmax(targets)``), the pre-migration behavior.
    """
    batch_size, seq_len, draft_vocab_size = logits.shape
    pos_idx = torch.arange(seq_len, device=logits.device) % block_size
    pos_idx = pos_idx.unsqueeze(0).expand(batch_size, -1)
    loss_mask = loss_mask.to(logits.dtype)
    loss_weights = loss_mask * dflash_loss_decay(pos_idx.to(logits.dtype), gamma=gamma)

    # target_ids (target model top-1) still drives the accuracy / accept-rate
    # metrics below and is the CE fallback when no ground-truth label is passed.
    target_ids = torch.argmax(targets, dim=-1)
    if ce_label_ids is None:
        ce_target_ids = target_ids
        ce_weights = loss_weights
    else:
        ce_target_ids = ce_label_ids
        ce_weights = loss_weights
        if ce_label_valid is not None:
            ce_weights = loss_weights * ce_label_valid.to(loss_weights.dtype)
    ce_per_token = F.cross_entropy(
        logits.reshape(-1, draft_vocab_size),
        ce_target_ids.reshape(-1),
        reduction="none",
    ).reshape(batch_size, seq_len)
    ce_loss, ce_num, ce_den = _weighted_mean(ce_per_token, ce_weights)

    draft_probs = torch.softmax(logits.float(), dim=-1)
    target_probs = torch.softmax(targets.float(), dim=-1)
    l1_per_token = (draft_probs - target_probs).abs().sum(dim=-1).to(logits.dtype)
    l1_loss, l1_num, l1_den = _weighted_mean(l1_per_token, loss_weights)

    accept_rate = (1.0 - 0.5 * l1_per_token.detach()).clamp_(0.0, 1.0)
    confidence_loss = logits.new_zeros(())
    confidence_num = logits.new_zeros(())
    confidence_den = logits.new_zeros(())
    confidence_abs_error_num = logits.new_zeros(())
    confidence_bias_num = logits.new_zeros(())
    if confidence_logits is not None:
        confidence_per_token = F.binary_cross_entropy_with_logits(
            confidence_logits.float(),
            accept_rate.float(),
            reduction="none",
        ).to(logits.dtype)
        confidence_loss, confidence_num, confidence_den = _weighted_mean(
            confidence_per_token,
            loss_weights,
        )
        with torch.no_grad():
            confidence_error = confidence_logits.float().sigmoid() - accept_rate.float()
            confidence_abs_error_num = (
                confidence_error.abs().to(logits.dtype) * loss_weights
            ).sum()
            confidence_bias_num = (
                confidence_error.to(logits.dtype) * loss_weights
            ).sum()

    loss = (
        ce_loss_alpha * ce_loss
        + l1_loss_alpha * l1_loss
        + confidence_head_alpha * confidence_loss
    )

    pred_ids = torch.argmax(logits, dim=-1)
    correct_per_pos, total_per_pos = compute_accuracy_multi_step(
        pred_ids, target_ids, loss_mask, pos_idx, block_size
    )

    metrics: dict[str, Any] = {
        "loss_sum": loss.detach().clone(),
        "loss_total": torch.tensor(1.0, device=logits.device),
        "ce_loss_sum": ce_num.detach(),
        "ce_loss_total": ce_den.detach(),
        "l1_loss_sum": l1_num.detach(),
        "l1_loss_total": l1_den.detach(),
        "full_acc_sum": correct_per_pos[1:].sum(),
        "full_acc_total": total_per_pos[1:].sum(),
    }
    if confidence_logits is not None:
        metrics.update(
            {
                "confidence_loss_sum": confidence_num.detach(),
                "confidence_loss_total": confidence_den.detach(),
                "confidence_abs_error_sum": confidence_abs_error_num.detach(),
                "confidence_abs_error_total": confidence_den.detach(),
                "confidence_bias_sum": confidence_bias_num.detach(),
                "confidence_bias_total": confidence_den.detach(),
            }
        )

    with torch.no_grad():
        for pos in range(1, block_size):
            pos_mask = (pos_idx == pos) & loss_mask.bool()
            pos_total = pos_mask.float().sum()
            metrics[f"position_{pos}_acc_sum"] = correct_per_pos[pos]
            metrics[f"position_{pos}_acc_total"] = total_per_pos[pos]
            metrics[f"accept_rate_position_{pos}_sum"] = (
                accept_rate.masked_select(pos_mask).sum()
            )
            metrics[f"accept_rate_position_{pos}_total"] = pos_total

        accept_blocks = (accept_rate * loss_mask).view(batch_size, -1, block_size)
        block_eval_mask = loss_mask.view(batch_size, -1, block_size).bool()
        valid_blocks = block_eval_mask[..., 1:].any(dim=-1)
        prefix_accept = accept_blocks[..., 1:].cumprod(dim=-1).sum(dim=-1)
        metrics["tau_probabilistic_sum"] = (prefix_accept + 1.0).masked_select(
            valid_blocks
        ).sum()
        metrics["tau_probabilistic_total"] = valid_blocks.float().sum()

    return loss, metrics
