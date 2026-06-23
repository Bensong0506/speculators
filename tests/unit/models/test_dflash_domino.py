"""Unit tests for DFlash Domino correction head."""

import torch
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from speculators import SpeculatorsConfig, VerifierConfig
from speculators.models.dflash import DFlashSpeculatorConfig
from speculators.models.dflash.core import DFlashDraftModel
from speculators.proposals.greedy import GreedyTokenProposalConfig


def _fill_nan_weights(model):
    with torch.no_grad():
        for param in model.parameters():
            if param.isnan().any():
                torch.nn.init.normal_(param, mean=0.0, std=0.02)


def _make_domino_model(draft_vocab_size: int = 32):
    qwen_config = Qwen3Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        _attn_implementation="eager",  # type: ignore[call-arg]
    )
    config = DFlashSpeculatorConfig(
        transformer_layer_config=qwen_config,
        draft_vocab_size=draft_vocab_size,
        block_size=4,
        max_anchors=2,
        aux_hidden_state_layer_ids=[0],
        mask_token_id=0,
        domino_enabled=True,
        domino_emb_dim=8,
        domino_gru_hidden_dim=12,
        domino_pure_draft_prefix_len=1,
        speculators_config=SpeculatorsConfig(
            algorithm="dflash",
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=3)],
            default_proposal_method="greedy",
            verifier=VerifierConfig(name_or_path=None, architectures=["Qwen3ForCausalLM"]),
        ),
    )
    model = DFlashDraftModel(config)
    _fill_nan_weights(model)
    if draft_vocab_size != qwen_config.vocab_size:
        d2t = torch.arange(draft_vocab_size, dtype=torch.long)
        t2d = torch.zeros(qwen_config.vocab_size, dtype=torch.bool)
        t2d[:draft_vocab_size] = True
        model.load_vocab_mappings(t2d=t2d, d2t=d2t)
    return model


def test_domino_head_preserves_shape_and_prefix_logits():
    model = _make_domino_model(draft_vocab_size=64)
    base_logits = torch.randn(1, 8, 64)
    hidden_states = torch.randn(1, 8, 16)
    target_ids = torch.randint(0, 64, (1, 8))

    logits = model._apply_domino_head(base_logits, hidden_states, target_ids)

    assert logits.shape == base_logits.shape
    # Anchor and first speculative token remain pure DFlash when prefix_len=1.
    assert torch.equal(logits[:, 0:2], base_logits[:, 0:2])
    assert torch.equal(logits[:, 4:6], base_logits[:, 4:6])


def test_domino_head_supports_reduced_draft_vocab():
    model = _make_domino_model(draft_vocab_size=32)
    base_logits = torch.randn(1, 8, 32)
    hidden_states = torch.randn(1, 8, 16)
    target_ids = torch.randint(0, 32, (1, 8))

    logits = model._apply_domino_head(base_logits, hidden_states, target_ids)

    assert logits.shape == base_logits.shape
