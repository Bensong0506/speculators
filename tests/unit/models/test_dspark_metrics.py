"""Unit tests for DSpark metrics and loss functions."""

import pytest
import torch

from speculators.models.dspark.markov_head import VanillaMarkovHead
from speculators.models.dspark.metrics import compute_dspark_metrics


def _ids_to_logits(ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
    logits = torch.zeros(*ids.shape, vocab_size)
    logits.scatter_(-1, ids.unsqueeze(-1), 100.0)
    return logits


def test_dspark_metrics_perfect_distribution_has_accept_rate_one():
    target_ids = torch.tensor([[0, 1, 2, 3]])
    logits = _ids_to_logits(target_ids, vocab_size=5)
    targets = _ids_to_logits(target_ids, vocab_size=5)
    confidence_logits = torch.full((1, 4), 10.0)
    loss_mask = torch.tensor([[0, 1, 1, 1]])

    loss, metrics = compute_dspark_metrics(
        logits,
        targets,
        confidence_logits,
        loss_mask,
        block_size=4,
    )

    assert torch.isfinite(loss)
    assert metrics["full_acc_sum"].item() == pytest.approx(3.0)
    assert metrics["full_acc_total"].item() == pytest.approx(3.0)
    assert metrics["accept_rate_position_1_sum"].item() == pytest.approx(1.0)
    assert metrics["accept_rate_position_1_total"].item() == pytest.approx(1.0)


def test_dspark_metrics_wrong_distribution_has_low_accept_rate():
    logits = _ids_to_logits(torch.tensor([[0, 1, 1, 1]]), vocab_size=5)
    targets = _ids_to_logits(torch.tensor([[0, 2, 2, 2]]), vocab_size=5)
    confidence_logits = torch.zeros(1, 4)
    loss_mask = torch.tensor([[0, 1, 1, 1]])

    _loss, metrics = compute_dspark_metrics(
        logits,
        targets,
        confidence_logits,
        loss_mask,
        block_size=4,
    )

    assert metrics["full_acc_sum"].item() == pytest.approx(0.0)
    assert metrics["full_acc_total"].item() == pytest.approx(3.0)
    assert metrics["accept_rate_position_1_sum"].item() == pytest.approx(
        0.0, abs=1e-4
    )


def test_vanilla_markov_head_shapes():
    head = VanillaMarkovHead(
        verifier_vocab_size=11,
        draft_vocab_size=7,
        markov_rank=3,
    )
    prev_embeddings, bias = head(torch.tensor([[1, 2, 3, 4]]))

    assert prev_embeddings.shape == (1, 4, 3)
    assert bias.shape == (1, 4, 7)


def test_ce_ground_truth_label_overrides_target_argmax():
    # Draft predicts token 1 everywhere; target top-1 is token 2; ground truth is
    # token 1. With ce_label_ids the CE label is the ground truth (matches the
    # draft), so CE should be lower than the target-argmax fallback.
    logits = _ids_to_logits(torch.tensor([[0, 1, 1, 1]]), vocab_size=5)
    targets = _ids_to_logits(torch.tensor([[0, 2, 2, 2]]), vocab_size=5)
    loss_mask = torch.tensor([[0, 1, 1, 1]])
    gt_labels = torch.tensor([[0, 1, 1, 1]])
    valid = torch.ones_like(gt_labels, dtype=torch.bool)

    loss_gt, _ = compute_dspark_metrics(
        logits, targets, None, loss_mask, block_size=4,
        ce_label_ids=gt_labels, ce_label_valid=valid,
    )
    loss_argmax, _ = compute_dspark_metrics(
        logits, targets, None, loss_mask, block_size=4,
    )
    assert loss_gt.item() < loss_argmax.item()


def test_ce_out_of_vocab_positions_are_masked():
    # Every supervised position is flagged out-of-vocab -> CE term drops out and
    # the total loss reduces to the L1 term only (finite, no NaN from a bad index).
    logits = _ids_to_logits(torch.tensor([[0, 1, 2, 3]]), vocab_size=5)
    targets = _ids_to_logits(torch.tensor([[0, 1, 2, 3]]), vocab_size=5)
    loss_mask = torch.tensor([[0, 1, 1, 1]])
    gt_labels = torch.zeros(1, 4, dtype=torch.long)
    valid = torch.zeros(1, 4, dtype=torch.bool)

    loss, metrics = compute_dspark_metrics(
        logits, targets, None, loss_mask, block_size=4,
        ce_loss_alpha=1.0, l1_loss_alpha=0.0, confidence_head_alpha=0.0,
        ce_label_ids=gt_labels, ce_label_valid=valid,
    )
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)
