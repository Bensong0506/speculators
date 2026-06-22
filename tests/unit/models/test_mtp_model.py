"""Unit tests for MTPDraftModel forward pass."""

import math

import torch

BATCH = 1
SEQ_LEN = 10


# ===== Forward output structure =====


def test_forward_output_structure(mtp_model, seed):
    """Verify logit shapes, loss, and per-step metrics in a single forward pass."""
    num_steps = mtp_model.config.num_speculative_steps
    hidden_size = mtp_model.config.hidden_size
    vocab_size = mtp_model.config.vocab_size
    input_ids = torch.randint(0, vocab_size, (BATCH, SEQ_LEN))
    hidden_states = torch.randn(BATCH, SEQ_LEN, hidden_size)
    with torch.no_grad():
        logits_list, total_loss, metrics = mtp_model(
            input_ids=input_ids, hidden_states=hidden_states
        )

    assert len(logits_list) == num_steps
    expected_len = SEQ_LEN - num_steps - 1
    for step in range(num_steps):
        assert logits_list[step].shape == (BATCH, expected_len, vocab_size)

    assert total_loss.dim() == 0
    assert torch.isfinite(total_loss)
    assert total_loss >= 0

    expected_keys = {f"loss_step_{k}" for k in range(num_steps)} | {
        "loss_sum",
        "loss_total",
    }
    assert set(metrics.keys()) == expected_keys
    for key in expected_keys:
        assert math.isfinite(metrics[key])


# ===== Loss masking =====


class TestLossMasking:
    def test_zero_mask_ignores_all_targets(self, mtp_model, seed):
        """All-zero loss_mask sets every target to -100. Loss returns 0.0
        (not NaN) because the denominator is clamped to min=1."""
        hidden_size = mtp_model.config.hidden_size
        vocab_size = mtp_model.config.vocab_size
        input_ids = torch.randint(0, vocab_size, (BATCH, SEQ_LEN))
        hidden_states = torch.randn(BATCH, SEQ_LEN, hidden_size)
        loss_mask = torch.zeros(BATCH, SEQ_LEN)
        with torch.no_grad():
            _, total_loss, _ = mtp_model(
                input_ids=input_ids,
                hidden_states=hidden_states,
                loss_mask=loss_mask,
            )
        assert total_loss == 0.0

    def test_partial_mask_changes_loss(self, mtp_model, seed):
        """Masking some positions should change the loss vs no mask."""
        hidden_size = mtp_model.config.hidden_size
        vocab_size = mtp_model.config.vocab_size
        input_ids = torch.randint(0, vocab_size, (BATCH, SEQ_LEN))
        hidden_states = torch.randn(BATCH, SEQ_LEN, hidden_size)
        with torch.no_grad():
            _, loss_no_mask, _ = mtp_model(
                input_ids=input_ids, hidden_states=hidden_states
            )
            mask = torch.ones(BATCH, SEQ_LEN)
            mask[:, -3:] = 0
            _, loss_partial_mask, _ = mtp_model(
                input_ids=input_ids, hidden_states=hidden_states, loss_mask=mask
            )
        assert loss_no_mask != loss_partial_mask


# ===== Step weights =====


class TestStepWeights:
    def test_zero_weight_zeroes_step_loss(self, mtp_model, seed):
        hidden_size = mtp_model.config.hidden_size
        vocab_size = mtp_model.config.vocab_size
        input_ids = torch.randint(0, vocab_size, (BATCH, SEQ_LEN))
        hidden_states = torch.randn(BATCH, SEQ_LEN, hidden_size)
        with torch.no_grad():
            _, _, metrics = mtp_model(
                input_ids=input_ids,
                hidden_states=hidden_states,
                step_weights=[1.0, 0.0, 0.0],
            )
        assert metrics["loss_step_0"] > 0
        assert metrics["loss_step_1"] == 0.0
        assert metrics["loss_step_2"] == 0.0


# ===== Self-forcing =====


class TestSelfForcing:
    def test_zero_probability_matches_default(self, mtp_model, seed):
        hidden_size = mtp_model.config.hidden_size
        vocab_size = mtp_model.config.vocab_size
        input_ids = torch.randint(0, vocab_size, (BATCH, SEQ_LEN))
        hidden_states = torch.randn(BATCH, SEQ_LEN, hidden_size)

        with torch.no_grad():
            default_logits, default_loss, _ = mtp_model(
                input_ids=input_ids,
                hidden_states=hidden_states,
            )
            forced_logits, forced_loss, _ = mtp_model(
                input_ids=input_ids,
                hidden_states=hidden_states,
                self_forcing_p=0.0,
            )

        assert len(default_logits) == len(forced_logits)
        for default_step_logits, forced_step_logits in zip(
            default_logits, forced_logits, strict=True
        ):
            torch.testing.assert_close(default_step_logits, forced_step_logits)
        torch.testing.assert_close(default_loss, forced_loss)

    def test_probability_one_feeds_previous_argmax(self, mtp_model, seed):
        hidden_size = mtp_model.config.hidden_size
        vocab_size = mtp_model.config.vocab_size
        input_ids = torch.randint(0, vocab_size, (BATCH, SEQ_LEN))
        hidden_states = torch.randn(BATCH, SEQ_LEN, hidden_size)
        captured_tokens = []

        def capture_embed_input(_module, args):
            captured_tokens.append(args[0].detach().clone())

        handle = mtp_model.embed_tokens.register_forward_pre_hook(capture_embed_input)
        try:
            with torch.no_grad():
                logits_list, _, _ = mtp_model(
                    input_ids=input_ids,
                    hidden_states=hidden_states,
                    self_forcing_p=1.0,
                )
        finally:
            handle.remove()

        effective_steps = len(logits_list)
        valid_len = SEQ_LEN - effective_steps - 1
        assert torch.equal(captured_tokens[0], input_ids[:, 1 : 1 + valid_len])
        for step in range(1, effective_steps):
            expected_tokens = logits_list[step - 1].argmax(dim=-1)
            assert torch.equal(captured_tokens[step], expected_tokens)


# ===== Short sequence truncation =====


def test_short_sequence_fewer_logits(mtp_model, seed):
    num_steps = mtp_model.config.num_speculative_steps
    hidden_size = mtp_model.config.hidden_size
    vocab_size = mtp_model.config.vocab_size
    short_len = 3
    input_ids = torch.randint(0, vocab_size, (BATCH, short_len))
    hidden_states = torch.randn(BATCH, short_len, hidden_size)
    with torch.no_grad():
        logits_list, _, _ = mtp_model(input_ids=input_ids, hidden_states=hidden_states)
    assert len(logits_list) < num_steps
    assert len(logits_list) == 1
