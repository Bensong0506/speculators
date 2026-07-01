from typing import ClassVar

import torch
from torch import nn
from transformers import PretrainedConfig

from speculators.model import SpeculatorModel
from speculators.models.dflash.core import DFlashDraftModel, maybe_compile_dflash_forward
from speculators.models.dflash.utils import get_base_indices_for_anchored_blocks
from speculators.models.dspark.config import DSparkSpeculatorConfig
from speculators.models.dspark.markov_head import VanillaMarkovHead
from speculators.models.dspark.metrics import compute_dspark_metrics
from speculators.models.utils import resolve_target_layer_ids


@SpeculatorModel.register("dspark")
class DSparkDraftModel(DFlashDraftModel):
    config_class: ClassVar[type[DSparkSpeculatorConfig]] = DSparkSpeculatorConfig  # type: ignore[misc]
    _no_split_modules = ["Qwen3DFlashDecoderLayer"]

    def __init__(
        self,
        config: DSparkSpeculatorConfig,
    ) -> None:
        super().__init__(config=config)

        self.markov_head = VanillaMarkovHead(
            verifier_vocab_size=self.verifier_vocab_size,
            draft_vocab_size=self.draft_vocab_size,
            markov_rank=config.markov_rank,
        )
        confidence_in_features = config.transformer_layer_config.hidden_size
        if config.confidence_head_with_markov:
            confidence_in_features += config.markov_rank
        self.confidence_head = nn.Linear(confidence_in_features, 1)

    @classmethod
    def from_training_args(
        cls,
        verifier_config: "PretrainedConfig",
        t2d: torch.Tensor | None = None,
        d2t: torch.Tensor | None = None,
        **kwargs,
    ) -> "DSparkDraftModel":
        from speculators.config import (  # noqa: PLC0415
            SpeculatorsConfig,
            VerifierConfig,
        )
        from speculators.proposals.greedy import (  # noqa: PLC0415
            GreedyTokenProposalConfig,
        )

        target_layer_ids = resolve_target_layer_ids(
            kwargs.get("target_layer_ids"), kwargs["verifier_name_or_path"]
        )
        block_size = kwargs.get("block_size", 8)
        config = DSparkSpeculatorConfig(
            transformer_layer_config=verifier_config,
            draft_vocab_size=kwargs["draft_vocab_size"],
            block_size=block_size,
            max_anchors=kwargs.get("max_anchors", 3072),
            aux_hidden_state_layer_ids=target_layer_ids,
            mask_token_id=kwargs.get("mask_token_id"),
            sliding_window_non_causal=kwargs.get("sliding_window_non_causal", False),
            markov_rank=kwargs.get("markov_rank", 256),
            confidence_head_alpha=kwargs.get("confidence_head_alpha", 1.0),
            confidence_head_with_markov=kwargs.get("confidence_head_with_markov", True),
            ce_loss_alpha=kwargs.get("ce_loss_alpha", 0.1),
            l1_loss_alpha=kwargs.get("l1_loss_alpha", 0.9),
            loss_decay_gamma=kwargs.get("loss_decay_gamma", 4.0),
            ce_target=kwargs.get("ce_target", "ground_truth"),
            speculators_config=SpeculatorsConfig(
                algorithm="dspark",
                proposal_methods=[
                    GreedyTokenProposalConfig(
                        # This repo's block_size includes the anchor at position 0.
                        speculative_tokens=block_size - 1,
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
        call_kwargs = {
            "ce_loss_alpha": kwargs["ce_loss_alpha"],
            "l1_loss_alpha": kwargs["l1_loss_alpha"],
            "confidence_head_alpha": kwargs["confidence_head_alpha"],
            "loss_decay_gamma": kwargs["loss_decay_gamma"],
        }
        return call_kwargs, call_kwargs

    def _get_prev_token_ids(
        self,
        input_ids: torch.Tensor,
        anchored_block_indices: torch.Tensor,
    ) -> torch.Tensor:
        block_indices = anchored_block_indices.view(-1, self.block_size)
        prev_indices = (block_indices - 1).clamp(min=0)
        prev_indices[:, 0] = block_indices[:, 0]
        return input_ids[:, prev_indices.reshape(-1)]

    def _build_ce_ground_truth_labels(
        self,
        input_ids: torch.Tensor,  # shape: [1, total_seq_len]
        anchored_block_indices: torch.Tensor,  # shape: [num_anchors*block_size]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Paper-faithful CE labels: the realized next token at each draft slot.

        ``anchored_block_indices[k]`` is the source-sequence position whose token
        draft slot ``k`` must predict, so the ground-truth (verifier-vocab) label
        ids are ``input_ids`` gathered at those same indices. When the draft uses
        a reduced vocabulary we map each label from the verifier vocab into the
        draft vocab via ``d2t`` (``verifier_id = draft_id + d2t[draft_id]``) and
        flag positions whose token is out-of-vocab so they can be masked out of
        the CE term. (Draft slot 0 is the anchor and is masked from the loss
        upstream, so its label value is irrelevant.)
        """
        gt_verifier_ids = input_ids[:, anchored_block_indices]
        if not self.use_draft_vocab:
            valid = torch.ones_like(gt_verifier_ids, dtype=torch.bool)
            return gt_verifier_ids, valid

        device = input_ids.device
        draft_ids = torch.arange(self.draft_vocab_size, device=device)
        verifier_of_draft = draft_ids + self.d2t.to(device=device)
        t2d_index = input_ids.new_full((self.verifier_vocab_size,), -1)
        t2d_index[verifier_of_draft] = draft_ids
        draft_labels = t2d_index[gt_verifier_ids]
        valid = draft_labels >= 0
        return draft_labels.clamp(min=0), valid

    @maybe_compile_dflash_forward
    def forward(
        self,
        hidden_states: torch.Tensor,  # shape: [1,total_seq_len,num_hidden*hidden_size]
        input_ids: torch.Tensor,  # shape: [1, total_seq_len]
        loss_mask: torch.Tensor,  # shape: [1, total_seq_len]
        verifier_last_hidden_states: torch.Tensor,  # shape: [1, total_seq_len, hidden_size] # noqa: E501
        lengths: torch.Tensor | None = None,  # shape: [batch_size]
        position_ids: torch.Tensor | None = None,  # shape: [1, total_seq_len]
        ce_loss_alpha: float | None = None,
        l1_loss_alpha: float | None = None,
        confidence_head_alpha: float | None = None,
        loss_decay_gamma: float | None = None,
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
        )
        mask_token_ids[:, :: self.block_size] = input_ids[:, anchor_positions]
        noise_embedding = self.embed_tokens(mask_token_ids)

        fc_output = self.fc(hidden_states)
        fc_output = self.hidden_norm(fc_output)

        mask_position_ids = get_base_indices_for_anchored_blocks(
            position_ids[:, anchor_positions], self.block_size, input_ids.numel()
        )
        position_ids = torch.cat([position_ids, mask_position_ids.unsqueeze(0)], dim=1)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        anchored_block_indices = get_base_indices_for_anchored_blocks(
            anchor_positions, self.block_size, input_ids.numel()
        )

        with torch.no_grad():
            verifier_logits = self.verifier_lm_head(
                self.verifier_norm(verifier_last_hidden_states)
            )
            verifier_logits = torch.roll(verifier_logits, 1, dims=1)
            targets = verifier_logits[:, anchored_block_indices]

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

        normalized_hidden = self.norm(noise_embedding)
        base_logits = self.lm_head(normalized_hidden)
        prev_token_ids = self._get_prev_token_ids(input_ids, anchored_block_indices)
        prev_embeddings, markov_bias = self.markov_head(prev_token_ids)
        logits = base_logits + markov_bias

        confidence_inputs = normalized_hidden
        if self.config.confidence_head_with_markov:
            confidence_inputs = torch.cat([confidence_inputs, prev_embeddings], dim=-1)
        confidence_logits = self.confidence_head(confidence_inputs).squeeze(-1)

        aligned_loss_mask = loss_mask.clone()[:, anchored_block_indices]
        aligned_loss_mask = aligned_loss_mask * (
            anchor_valid.repeat_interleave(self.block_size)
            .unsqueeze(0)
            .to(aligned_loss_mask.dtype)
        )
        aligned_loss_mask[:, :: self.block_size] = 0

        ce_label_ids = None
        ce_label_valid = None
        if self.config.ce_target == "ground_truth":
            ce_label_ids, ce_label_valid = self._build_ce_ground_truth_labels(
                input_ids, anchored_block_indices
            )

        loss, metrics = compute_dspark_metrics(
            logits,
            targets,
            confidence_logits,
            aligned_loss_mask,
            self.block_size,
            ce_label_ids=ce_label_ids,
            ce_label_valid=ce_label_valid,
            ce_loss_alpha=(
                self.config.ce_loss_alpha if ce_loss_alpha is None else ce_loss_alpha
            ),
            l1_loss_alpha=(
                self.config.l1_loss_alpha if l1_loss_alpha is None else l1_loss_alpha
            ),
            confidence_head_alpha=(
                self.config.confidence_head_alpha
                if confidence_head_alpha is None
                else confidence_head_alpha
            ),
            gamma=(
                self.config.loss_decay_gamma
                if loss_decay_gamma is None
                else loss_decay_gamma
            ),
        )
        draft_tokens = torch.argmax(logits, dim=-1)

        return draft_tokens, loss, metrics
