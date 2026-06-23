import os
from typing import ClassVar

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask
from transformers import PretrainedConfig
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)

from speculators.model import DraftVocabMixin, SpeculatorModel
from speculators.models.dflash import DFlashSpeculatorConfig
from speculators.models.dflash.attention import create_anchor_block_mask_mod
from speculators.models.dflash.metrics import compute_metrics
from speculators.models.dflash.model_definitions import Qwen3DFlashDecoderLayer
from speculators.models.dflash.utils import (
    get_base_indices_for_anchored_blocks,
    select_anchors,
)
from speculators.models.metrics import (
    ce_loss,
    dflash_loss_decay,
    kl_div_loss,
    resolve_loss_fn,
)
from speculators.models.utils import resolve_target_layer_ids


def maybe_compile_dflash_forward(func):
    compile_enabled = os.environ.get("SPECULATORS_DFLASH_COMPILE", "1").lower()
    if compile_enabled in {"0", "false", "no", "off"}:
        return func
    return torch.compile(func)


@SpeculatorModel.register("dflash")
class DFlashDraftModel(DraftVocabMixin, SpeculatorModel):
    config_class: ClassVar[type[DFlashSpeculatorConfig]] = DFlashSpeculatorConfig  # type: ignore[misc]
    _no_split_modules = ["Qwen3DFlashDecoderLayer"]
    _keys_to_ignore_on_load_missing: ClassVar[list[str]] = [  # type: ignore[misc]
        "embed_tokens.weight",
        "verifier_norm.weight",
        "t2d",
        "d2t",
    ]
    _keys_to_ignore_on_save: ClassVar[list[str]] = [  # type: ignore[misc,assignment]
        "verifier_lm_head.weight",
        "verifier_norm.weight",
    ]

    t2d: torch.Tensor | None
    d2t: torch.Tensor | None

    def __init__(
        self,
        config: DFlashSpeculatorConfig,
    ) -> None:
        # Forcibly override config settings
        if config.transformer_layer_config._attn_implementation is None:  # noqa: SLF001
            config.transformer_layer_config._attn_implementation = (  # noqa: SLF001
                "simple_flex_attention"
            )
        super().__init__(config=config)
        self._init_vocab(config)

        tl_config = config.transformer_layer_config

        # Number of draft layers is encoded in transformer_layer_config
        num_draft_layers = tl_config.num_hidden_layers
        self.layers = nn.ModuleList(
            [
                Qwen3DFlashDecoderLayer(config.transformer_layer_config, layer_idx)  # type: ignore[arg-type]
                for layer_idx in range(num_draft_layers)
            ]
        )
        self.sliding_window = tl_config.sliding_window
        self.sliding_window_indices = [
            i
            for i, layer_type in enumerate(tl_config.layer_types)
            if layer_type == "sliding_attention"
        ]
        self.uses_sliding_window_attn = bool(self.sliding_window_indices)
        self.uses_full_attn = bool(num_draft_layers - len(self.sliding_window_indices))
        self.sliding_window_non_causal = config.sliding_window_non_causal

        if config.aux_hidden_state_layer_ids is None:
            raise ValueError(
                "aux_hidden_state_layer_ids must be set in DFlashSpeculatorConfig. "
                "Use DFlashDraftModel.from_training_args() to resolve defaults."
            )
        self.target_layer_ids = config.aux_hidden_state_layer_ids

        self.norm = Qwen3RMSNorm(
            config.transformer_layer_config.hidden_size,
            eps=config.transformer_layer_config.rms_norm_eps,  # type: ignore[arg-type]
        )
        self.rotary_emb = Qwen3RotaryEmbedding(config.transformer_layer_config)  # type: ignore[arg-type]

        self.fc = nn.Linear(
            len(self.target_layer_ids) * config.transformer_layer_config.hidden_size,
            config.transformer_layer_config.hidden_size,
            bias=False,
        )
        self.hidden_norm = Qwen3RMSNorm(
            config.transformer_layer_config.hidden_size,
            eps=config.transformer_layer_config.rms_norm_eps,  # type: ignore[arg-type]
        )
        self.domino_enabled = config.domino_enabled
        self.domino_pure_draft_prefix_len = config.domino_pure_draft_prefix_len
        if self.domino_enabled:
            if self.domino_pure_draft_prefix_len < 0:
                raise ValueError("domino_pure_draft_prefix_len must be >= 0")
            if self.domino_pure_draft_prefix_len >= config.block_size:
                raise ValueError(
                    "domino_pure_draft_prefix_len must be smaller than block_size"
                )
            self.prefix_gru = nn.GRU(
                input_size=config.transformer_layer_config.hidden_size,
                hidden_size=config.domino_gru_hidden_dim,
                num_layers=1,
                batch_first=True,
                bias=False,
            )
            self.embed_proj = nn.Sequential(
                nn.Linear(
                    config.transformer_layer_config.hidden_size
                    + config.domino_gru_hidden_dim,
                    config.domino_emb_dim,
                    bias=False,
                ),
                nn.SiLU(),
                nn.Linear(config.domino_emb_dim, config.draft_vocab_size, bias=False),
            )
        self.verifier_norm = Qwen3RMSNorm(
            config.transformer_layer_config.hidden_size,
            eps=config.transformer_layer_config.rms_norm_eps,  # type: ignore[arg-type]
        )
        self.verifier_norm.weight.requires_grad = False
        self.block_size = config.block_size
        self.post_init()

    @classmethod
    def from_training_args(
        cls,
        verifier_config: "PretrainedConfig",
        t2d: torch.Tensor | None = None,
        d2t: torch.Tensor | None = None,
        **kwargs,
    ) -> "DFlashDraftModel":
        """Create DFlash model from training arguments.

        Args:
            verifier_config: Verifier model configuration. This should be a config
                with num_hidden_layers set to the number of DRAFT layers (created
                by create_transformer_layer_config in train.py).
            t2d: Target-to-draft vocabulary mapping tensor (optional)
            d2t: Draft-to-target vocabulary mapping tensor (optional)
            **kwargs: Training arguments with DFlash-specific params
                - draft_vocab_size: Size of draft vocabulary
                - block_size: Block size for draft predictions (default: 8)
                - max_anchors: Max anchor positions during training (default: 256)
                - verifier_name_or_path: Path to verifier model

        Returns:
            Initialized DFlashDraftModel

        Note:
            The number of draft layers is encoded in verifier_config.num_hidden_layers,
            following the same pattern as EAGLE3.
        """
        from speculators.config import (  # noqa: PLC0415
            SpeculatorsConfig,
            VerifierConfig,
        )
        from speculators.proposals.greedy import (  # noqa: PLC0415
            GreedyTokenProposalConfig,
        )

        target_layer_ids = resolve_target_layer_ids(
            kwargs.get("target_layer_ids"),
            kwargs["verifier_name_or_path"],
        )

        config = DFlashSpeculatorConfig(
            transformer_layer_config=verifier_config,
            draft_vocab_size=kwargs["draft_vocab_size"],
            block_size=kwargs.get("block_size", 8),
            max_anchors=kwargs.get("max_anchors", 3072),
            aux_hidden_state_layer_ids=target_layer_ids,
            mask_token_id=kwargs.get("mask_token_id"),
            sliding_window_non_causal=kwargs.get("sliding_window_non_causal", False),
            domino_enabled=kwargs.get("dflash_domino", False),
            domino_emb_dim=kwargs.get("domino_emb_dim", 256),
            domino_gru_hidden_dim=kwargs.get("domino_gru_hidden_dim", 1024),
            domino_pure_draft_prefix_len=kwargs.get(
                "domino_pure_draft_prefix_len", 1
            ),
            speculators_config=SpeculatorsConfig(
                algorithm="dflash",
                proposal_methods=[
                    GreedyTokenProposalConfig(
                        # DFlash first position is anchor position, not used during gen
                        speculative_tokens=kwargs.get("block_size", 8) - 1,
                    )
                ],
                default_proposal_method="greedy",
                verifier=VerifierConfig.from_config(
                    verifier_config, name_or_path=kwargs["verifier_name_or_path"]
                ),
            ),
        )

        model = cls(config=config)
        model.load_vocab_mappings(t2d, d2t)
        model.load_verifier_weights()
        return model

    @staticmethod
    def get_trainer_kwargs(**kwargs) -> tuple[dict, dict]:
        """Get training and validation kwargs for DFlash.

        Args:
            **kwargs: Training arguments

        Returns:
            Tuple of (train_call_kwargs, val_call_kwargs)
        """
        loss_fn = resolve_loss_fn(kwargs["loss_fn"])
        train_kwargs = {"loss_fn": loss_fn}
        if kwargs.get("dflash_domino", False):
            train_kwargs.update(
                {
                    "domino_lambda_base_start": kwargs.get(
                        "domino_lambda_base_start", 1.0
                    ),
                    "domino_lambda_base_decay_ratio": kwargs.get(
                        "domino_lambda_base_decay_ratio", 1.0
                    ),
                    "domino_loss_decay_gamma": kwargs.get("domino_loss_decay_gamma"),
                }
            )
        val_kwargs = {
            "loss_fn": loss_fn,
            "domino_lambda_base": 0.0,
            "domino_loss_decay_gamma": kwargs.get("domino_loss_decay_gamma"),
        }
        return train_kwargs, val_kwargs

    @property
    def mask_token_id(self) -> int:
        if self.config.mask_token_id is None:
            raise ValueError(
                "mask_token_id is not set on the config. "
                "Pass --mask-token-id during training or ensure the config "
                "was saved with mask_token_id set."
            )
        return self.config.mask_token_id

    @property
    def _domino_suffix_start(self) -> int:
        # Position 0 is the anchor and has no loss in the vLLM DFlash convention.
        return 1 + self.domino_pure_draft_prefix_len

    def _target_ids_to_embedding_ids(self, target_ids: torch.Tensor) -> torch.Tensor:
        if not self.use_draft_vocab:
            return target_ids
        if self.d2t is None:
            raise ValueError("d2t mapping is required for Domino with draft vocab")
        return self.d2t[target_ids]

    def _apply_domino_head(
        self,
        base_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        if not self.domino_enabled:
            return base_logits

        batch_size, flat_seq_len, hidden_size = hidden_states.shape
        block_size = self.block_size
        num_blocks = flat_seq_len // block_size
        suffix_start = self._domino_suffix_start
        if suffix_start >= block_size:
            return base_logits

        hidden4d = hidden_states.reshape(batch_size, num_blocks, block_size, hidden_size)
        target_ids4d = target_ids.reshape(batch_size, num_blocks, block_size)
        embed_ids4d = self._target_ids_to_embedding_ids(target_ids4d)
        block_emb = self.embed_tokens(embed_ids4d)

        gru_inputs = block_emb[:, :, : block_size - 1, :].reshape(
            batch_size * num_blocks, block_size - 1, hidden_size
        )
        gru_out, _ = self.prefix_gru(gru_inputs)
        gru_out = gru_out.reshape(
            batch_size, num_blocks, block_size - 1, self.config.domino_gru_hidden_dim
        )
        prefix_states = gru_out[:, :, suffix_start - 1 :, :]

        z_n = hidden4d[:, :, suffix_start:, :]
        correction_logits = self.embed_proj(torch.cat([z_n, prefix_states], dim=-1))

        base_logits4d = base_logits.reshape(
            batch_size, num_blocks, block_size, base_logits.shape[-1]
        )
        prefix_logits = base_logits4d[:, :, :suffix_start, :]
        suffix_logits = base_logits4d[:, :, suffix_start:, :] + correction_logits
        return torch.cat([prefix_logits, suffix_logits], dim=2).reshape_as(base_logits)

    def _compute_domino_loss_and_metrics(
        self,
        logits: torch.Tensor,
        base_logits: torch.Tensor,
        targets: torch.Tensor,
        loss_mask: torch.Tensor,
        lambda_base: float,
        loss_decay_gamma: float | None,
    ) -> tuple[torch.Tensor, dict]:
        pos_idx = torch.arange(logits.shape[1], device=logits.device) % self.block_size
        pos_idx = pos_idx.unsqueeze(0)
        weight_mask = loss_mask.to(logits.dtype)
        if loss_decay_gamma is not None and loss_decay_gamma > 0:
            weight_mask = weight_mask * dflash_loss_decay(
                pos_idx.to(logits.dtype), loss_decay_gamma
            )

        target_ids = torch.argmax(targets, dim=-1)
        denom = weight_mask.sum() + 1e-5
        final_loss_per_token = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target_ids.reshape(-1),
            reduction="none",
        ).reshape_as(weight_mask)
        base_loss_per_token = F.cross_entropy(
            base_logits.reshape(-1, base_logits.shape[-1]),
            target_ids.reshape(-1),
            reduction="none",
        ).reshape_as(weight_mask)
        final_loss = (final_loss_per_token * weight_mask).sum() / denom
        base_loss = (base_loss_per_token * weight_mask).sum() / denom
        loss = (1.0 - lambda_base) * final_loss + lambda_base * base_loss

        _, metrics = compute_metrics(
            logits, targets, loss_mask, self.block_size, loss_fn=ce_loss
        )
        with torch.no_grad():
            base_pred_ids = torch.argmax(base_logits, dim=-1)
            final_pred_ids = torch.argmax(logits, dim=-1)
            valid = loss_mask.to(torch.bool)
            actual = valid.sum().to(logits.dtype) + 1e-5
            base_acc = ((base_pred_ids == target_ids) & valid).sum().to(logits.dtype)
            final_acc = ((final_pred_ids == target_ids) & valid).sum().to(logits.dtype)
            metrics["domino_final_loss_sum"] = final_loss.detach()
            metrics["domino_final_loss_total"] = torch.tensor(1.0, device=logits.device)
            metrics["domino_base_loss_sum"] = base_loss.detach()
            metrics["domino_base_loss_total"] = torch.tensor(1.0, device=logits.device)
            metrics["domino_base_acc_sum"] = base_acc
            metrics["domino_base_acc_total"] = actual
            metrics["domino_final_acc_sum"] = final_acc
            metrics["domino_final_acc_total"] = actual
            metrics["domino_lambda_base"] = torch.tensor(lambda_base, device=logits.device)
        return loss, metrics

    @torch.compiler.disable
    def _build_attention_mask(self, loss_mask, lengths, device):
        total_seq_len = loss_mask.shape[1]

        anchor_positions, anchor_valid = select_anchors(
            loss_mask, self.config.max_anchors, self.block_size
        )

        full_attn_mask = None
        if self.uses_full_attn:
            mask_mod, q_len, kv_len = create_anchor_block_mask_mod(
                lengths=lengths.to(device),
                total_seq_len=total_seq_len,
                anchor_positions=anchor_positions,
                block_size=self.block_size,
                sliding_window=None,
            )
            full_attn_mask = create_block_mask(
                mask_mod, B=None, H=None, Q_LEN=q_len, KV_LEN=kv_len, device=device
            )

        sliding_window_attn_mask = None
        if self.uses_sliding_window_attn:
            mask_mod, q_len, kv_len = create_anchor_block_mask_mod(
                lengths=lengths.to(device),
                total_seq_len=total_seq_len,
                anchor_positions=anchor_positions,
                block_size=self.block_size,
                sliding_window=self.sliding_window,
                sliding_window_non_causal=self.sliding_window_non_causal,
            )
            sliding_window_attn_mask = create_block_mask(
                mask_mod, B=None, H=None, Q_LEN=q_len, KV_LEN=kv_len, device=device
            )

        return full_attn_mask, sliding_window_attn_mask, anchor_positions, anchor_valid

    @maybe_compile_dflash_forward
    def forward(
        self,
        hidden_states: torch.Tensor,  # shape: [1,total_seq_len,num_hidden*hidden_size]
        input_ids: torch.Tensor,  # shape: [1, total_seq_len]
        loss_mask: torch.Tensor,  # shape: [1, total_seq_len]
        verifier_last_hidden_states: torch.Tensor,  # shape: [1, total_seq_len, hidden_size] # noqa: E501
        lengths: torch.Tensor | None = None,  # shape: [batch_size]
        position_ids: torch.Tensor | None = None,  # shape: [1, total_seq_len]
        loss_fn=kl_div_loss,
        domino_lambda_base: float = 0.0,
        domino_loss_decay_gamma: float | None = None,
        **kwargs,
    ):
        device = hidden_states.device
        total_seq_len = hidden_states.shape[1]
        num_anchors = self.config.max_anchors

        if lengths is None:
            lengths = torch.tensor([total_seq_len], dtype=torch.long, device=device)
        if position_ids is None:
            position_ids = 1 + torch.arange(
                total_seq_len, dtype=torch.long, device=device
            ).unsqueeze(0)

        full_attn_mask, sliding_window_attn_mask, anchor_positions, anchor_valid = (
            self._build_attention_mask(loss_mask, lengths, device)
        )

        mask_tokens_size = num_anchors * self.block_size

        mask_token_ids = torch.full(
            (1, mask_tokens_size),
            self.mask_token_id,
            dtype=torch.long,
            device=device,
        )  # shape: [1, num_anchors*block_size]
        mask_token_ids[:, :: self.block_size] = input_ids[:, anchor_positions]
        noise_embedding = self.embed_tokens(mask_token_ids)
        # shape: [1, num_anchors*block_size, hidden_size]

        fc_output = self.fc(hidden_states)
        fc_output = self.hidden_norm(fc_output)
        # shape: [1, total_seq_len, hidden_size]

        mask_position_ids = get_base_indices_for_anchored_blocks(
            position_ids[:, anchor_positions], self.block_size, input_ids.numel()
        )
        position_ids = torch.cat([position_ids, mask_position_ids.unsqueeze(0)], dim=1)
        # shape: [1, total_seq_len + num_anchors*block_size]

        # the hidden_states shape doesn't match position_ids but doesn't need
        # to, as hidden_states is only used to set dtype and device in rotary_emb
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        anchored_block_indices = get_base_indices_for_anchored_blocks(
            anchor_positions, self.block_size, input_ids.numel()
        )  # shape: [num_anchors*block_size]

        with torch.no_grad():
            verifier_logits = self.verifier_lm_head(
                self.verifier_norm(verifier_last_hidden_states)
            )
            # Shift right by 1 so verifier_logits[i] predicts token at position i
            verifier_logits = torch.roll(verifier_logits, 1, dims=1)
            targets = verifier_logits[:, anchored_block_indices]
            # shape: [1, num_anchors*block_size, draft_vocab_size]

        for layer_idx, layer in enumerate(self.layers):
            noise_embedding = layer(
                hidden_states=noise_embedding,
                target_hidden=fc_output,
                attention_mask=sliding_window_attn_mask
                if layer_idx in self.sliding_window_indices
                else full_attn_mask,
                position_ids=position_ids,
                use_cache=False,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        model_hidden = self.norm(noise_embedding)
        base_logits = self.lm_head(model_hidden)
        target_ids = torch.argmax(targets, dim=-1)
        logits = self._apply_domino_head(base_logits, model_hidden, target_ids)
        # shape: [1, num_anchors*block_size, vocab_size]

        aligned_loss_mask = loss_mask.clone()[:, anchored_block_indices]
        # shape: [1, num_anchors*block_size]

        # zero out any padded anchor blocks
        aligned_loss_mask = aligned_loss_mask * (
            anchor_valid.repeat_interleave(self.block_size)
            .unsqueeze(0)
            .to(aligned_loss_mask.dtype)
        )  # shape: [1, num_anchors*block_size]

        aligned_loss_mask[:, :: self.block_size] = 0
        if self.domino_enabled:
            loss, metrics = self._compute_domino_loss_and_metrics(
                logits=logits,
                base_logits=base_logits,
                targets=targets,
                loss_mask=aligned_loss_mask,
                lambda_base=domino_lambda_base,
                loss_decay_gamma=domino_loss_decay_gamma,
            )
        else:
            loss, metrics = compute_metrics(
                logits, targets, aligned_loss_mask, self.block_size, loss_fn=loss_fn
            )
        draft_tokens = torch.argmax(logits, dim=-1)

        return draft_tokens, loss, metrics
