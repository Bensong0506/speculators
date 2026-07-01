"""Unit tests for the DSpark sequential head variants (need torch)."""

import torch

from speculators.models.dspark.markov_head import (
    GatedMarkovHead,
    RNNHead,
    VanillaMarkovHead,
    build_markov_head,
)


def _make(head_type, *, v=11, d=7, r=3, h=5):
    return build_markov_head(
        markov_head_type=head_type,
        verifier_vocab_size=v,
        draft_vocab_size=d,
        markov_rank=r,
        hidden_size=h,
    )


def test_build_markov_head_types():
    assert isinstance(_make("vanilla"), VanillaMarkovHead)
    assert isinstance(_make("gated"), GatedMarkovHead)
    assert isinstance(_make("rnn"), RNNHead)


def test_vanilla_block_matches_manual_bias():
    head = _make("vanilla")
    base = torch.randn(1, 2, 4, 7)  # [B, num_blocks, block_size, draft_vocab]
    token_ids = torch.randint(0, 11, (1, 2, 4))
    hidden = torch.randn(1, 2, 4, 5)
    out = head.apply_block_logits(base, token_ids=token_ids, hidden_states=hidden)
    manual = base + head.project_bias(head.get_prev_embeddings(token_ids))
    assert out.shape == base.shape
    assert torch.allclose(out, manual, atol=1e-5)


def test_gated_and_rnn_block_shapes():
    for head_type in ("gated", "rnn"):
        head = _make(head_type)
        base = torch.randn(1, 2, 4, 7)
        token_ids = torch.randint(0, 11, (1, 2, 4))
        hidden = torch.randn(1, 2, 4, 5)
        out = head.apply_block_logits(base, token_ids=token_ids, hidden_states=hidden)
        assert out.shape == base.shape


def test_rnn_step_carries_state():
    head = _make("rnn")
    state = head.init_recurrent_state(1, device=torch.device("cpu"), dtype=torch.float32)
    assert state is not None and state.shape == (1, 3)
    base_step = torch.randn(1, 7)
    token_ids = torch.zeros(1, dtype=torch.long)
    hidden = torch.randn(1, 5)
    logits1, state1 = head.step_logits(
        base_step, token_ids=token_ids, hidden_states=hidden, state=state
    )
    assert logits1.shape == (1, 7)
    # The state must change after a step (recurrence is active).
    assert not torch.allclose(state1, state)


def test_vanilla_step_has_no_state():
    head = _make("vanilla")
    assert head.init_recurrent_state(
        1, device=torch.device("cpu"), dtype=torch.float32
    ) is None
    base_step = torch.randn(1, 7)
    logits, state = head.step_logits(
        base_step, token_ids=torch.zeros(1, dtype=torch.long)
    )
    assert logits.shape == (1, 7)
    assert state is None
