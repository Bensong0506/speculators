import torch
from torch import nn

__all__ = [
    "VanillaMarkovHead",
    "GatedMarkovHead",
    "RNNHead",
    "build_markov_head",
]


class VanillaMarkovHead(nn.Module):
    """Low-rank Markov correction head used by DSpark.

    For each draft position, the backbone emits base logits U_k. The Markov
    head looks at the previous token x_{k-1} and adds a low-rank bias
    ``W2(W1[x_{k-1}])`` to U_k. ``W1`` (``prev_token_embed``) is keyed by the
    full verifier vocabulary (previous tokens are real verifier ids) while
    ``W2`` (``bias_proj``) projects into the reduced draft vocabulary so the
    bias can be added directly to the draft ``base_logits``.

    The base head is memoryless (position k depends only on x_{k-1}) so its
    per-position bias can be computed for a whole block at once.
    """

    markov_head_type = "vanilla"

    def __init__(
        self,
        *,
        verifier_vocab_size: int,
        draft_vocab_size: int,
        markov_rank: int,
        hidden_size: int | None = None,
    ) -> None:
        super().__init__()
        if markov_rank <= 0:
            raise ValueError(f"markov_rank must be positive, got {markov_rank}")
        self.verifier_vocab_size = verifier_vocab_size
        self.draft_vocab_size = draft_vocab_size
        self.markov_rank = markov_rank
        self.hidden_size = hidden_size
        self.prev_token_embed = nn.Embedding(verifier_vocab_size, markov_rank)
        self.bias_proj = nn.Linear(markov_rank, draft_vocab_size, bias=False)

    def get_prev_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.prev_token_embed(token_ids.long())

    def project_bias(self, latent_states: torch.Tensor) -> torch.Tensor:
        return self.bias_proj(latent_states)

    def compute_step_bias(
        self,
        token_ids: torch.Tensor,  # shape: [..., ]
        hidden_states: torch.Tensor | None = None,  # shape: [..., hidden]
    ) -> torch.Tensor:  # shape: [..., draft_vocab]
        del hidden_states  # memoryless: base head ignores the backbone hidden
        return self.project_bias(self.get_prev_embeddings(token_ids))

    def apply_block_logits(
        self,
        base_logits: torch.Tensor,  # shape: [B, num_blocks, block_size, draft_vocab]
        *,
        token_ids: torch.Tensor,  # shape: [B, num_blocks, block_size]
        hidden_states: torch.Tensor | None,  # shape: [B, num_blocks, block_size, hid]
    ) -> torch.Tensor:
        if base_logits.size(-2) == 0:
            return base_logits
        return base_logits + self.compute_step_bias(token_ids, hidden_states)

    def init_recurrent_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        """Recurrent state carried across positions at inference.

        Memoryless heads (vanilla/gated) have no state, so this returns ``None``.
        """
        del batch_size, device, dtype
        return None

    def step_logits(
        self,
        base_logits_step: torch.Tensor,  # shape: [batch, draft_vocab]
        *,
        token_ids: torch.Tensor,  # shape: [batch] previously sampled token id
        hidden_states: torch.Tensor | None = None,  # shape: [batch, hidden]
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """One semi-autoregressive decode step for a serving engine.

        Corrects the backbone ``base_logits`` for the current position and
        returns ``(corrected_logits, new_state)``. Sampling the next token is
        left to the caller, which feeds the sampled token back as ``token_ids``
        (and ``new_state`` back as ``state``) for the next position.
        """
        del state
        corrected = base_logits_step + self.compute_step_bias(token_ids, hidden_states)
        return corrected, None

    def forward(self, prev_token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        prev_embeddings = self.get_prev_embeddings(prev_token_ids)
        return prev_embeddings, self.project_bias(prev_embeddings)


class GatedMarkovHead(VanillaMarkovHead):
    """Markov head whose low-rank bias is gated by the backbone hidden state.

    Still memoryless (position k only sees x_{k-1} and h_k), so it can also be
    applied to a whole block at once. The gate ``sigma(W_g [h_k; W1[x_{k-1}]])``
    modulates the previous-token embedding before the low-rank projection.
    """

    markov_head_type = "gated"

    def __init__(
        self,
        *,
        verifier_vocab_size: int,
        draft_vocab_size: int,
        markov_rank: int,
        hidden_size: int,
    ) -> None:
        super().__init__(
            verifier_vocab_size=verifier_vocab_size,
            draft_vocab_size=draft_vocab_size,
            markov_rank=markov_rank,
            hidden_size=hidden_size,
        )
        self.gate_proj = nn.Linear(hidden_size + markov_rank, markov_rank)

    def compute_step_bias(
        self,
        token_ids: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert hidden_states is not None, "GatedMarkovHead requires hidden_states"
        prev_embeddings = self.get_prev_embeddings(token_ids)
        gate_inputs = torch.cat(
            [hidden_states.to(prev_embeddings.dtype), prev_embeddings], dim=-1
        )
        gate = torch.sigmoid(self.gate_proj(gate_inputs)).to(prev_embeddings.dtype)
        return self.project_bias(gate * prev_embeddings)


class RNNHead(VanillaMarkovHead):
    """Recurrent sequential head (paper Eq. 6).

    Unlike the memoryless Markov heads, position k can access the full prefix
    history x_{<k} through a GRU-like recurrent state. ``joint_proj`` maps the
    concatenation ``[s_{k-1}; W1[x_{k-1}]; h_k]`` into gate, candidate and
    output components (``W_g, W_c, W_o`` from the paper), and the output branch
    produces the low-rank bias. Because the recurrence is sequential, the block
    is unrolled left-to-right during training (teacher-forced token_ids).
    """

    markov_head_type = "rnn"

    def __init__(
        self,
        *,
        verifier_vocab_size: int,
        draft_vocab_size: int,
        markov_rank: int,
        hidden_size: int,
    ) -> None:
        super().__init__(
            verifier_vocab_size=verifier_vocab_size,
            draft_vocab_size=draft_vocab_size,
            markov_rank=markov_rank,
            hidden_size=hidden_size,
        )
        # Joint [s_{k-1}; W1[x_{k-1}]; h_k] -> [gate; candidate; output], each R^r.
        self.joint_proj = nn.Linear(2 * markov_rank + hidden_size, 3 * markov_rank)

    def _rnn_step(
        self,
        state: torch.Tensor,  # shape: [*, r]
        prev_embeddings: torch.Tensor,  # shape: [*, r]
        hidden_states: torch.Tensor,  # shape: [*, hidden]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.cat(
            [state, prev_embeddings, hidden_states.to(prev_embeddings.dtype)], dim=-1
        )
        gate_raw, candidate_raw, output_raw = self.joint_proj(z).chunk(3, dim=-1)
        gate = torch.sigmoid(gate_raw)
        candidate = torch.tanh(candidate_raw)
        new_state = gate * state + (1.0 - gate) * candidate
        bias = self.project_bias(torch.tanh(output_raw))
        return new_state, bias

    def apply_block_logits(
        self,
        base_logits: torch.Tensor,  # shape: [B, num_blocks, block_size, draft_vocab]
        *,
        token_ids: torch.Tensor,  # shape: [B, num_blocks, block_size]
        hidden_states: torch.Tensor | None,  # shape: [B, num_blocks, block_size, hid]
    ) -> torch.Tensor:
        assert hidden_states is not None, "RNNHead requires hidden_states"
        block_size = base_logits.size(-2)
        if block_size == 0:
            return base_logits
        leading_shape = base_logits.shape[:-2]  # [B, num_blocks]
        state = torch.zeros(
            *leading_shape,
            self.markov_rank,
            device=base_logits.device,
            dtype=hidden_states.dtype,
        )
        output_logits = []
        for k in range(block_size):
            prev_emb = self.get_prev_embeddings(token_ids[..., k])
            state, bias = self._rnn_step(state, prev_emb, hidden_states[..., k, :])
            output_logits.append(base_logits[..., k, :] + bias)
        return torch.stack(output_logits, dim=-2)

    def init_recurrent_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.zeros(batch_size, self.markov_rank, device=device, dtype=dtype)

    def step_logits(
        self,
        base_logits_step: torch.Tensor,
        *,
        token_ids: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert hidden_states is not None, "RNNHead requires hidden_states"
        assert state is not None, "RNNHead.step_logits requires a recurrent state"
        prev_emb = self.get_prev_embeddings(token_ids)
        new_state, bias = self._rnn_step(state, prev_emb, hidden_states)
        return base_logits_step + bias, new_state


def build_markov_head(
    *,
    markov_head_type: str,
    verifier_vocab_size: int,
    draft_vocab_size: int,
    markov_rank: int,
    hidden_size: int,
) -> VanillaMarkovHead:
    head_type = str(markov_head_type).lower()
    if head_type == "vanilla":
        return VanillaMarkovHead(
            verifier_vocab_size=verifier_vocab_size,
            draft_vocab_size=draft_vocab_size,
            markov_rank=markov_rank,
        )
    if head_type == "gated":
        return GatedMarkovHead(
            verifier_vocab_size=verifier_vocab_size,
            draft_vocab_size=draft_vocab_size,
            markov_rank=markov_rank,
            hidden_size=hidden_size,
        )
    if head_type == "rnn":
        return RNNHead(
            verifier_vocab_size=verifier_vocab_size,
            draft_vocab_size=draft_vocab_size,
            markov_rank=markov_rank,
            hidden_size=hidden_size,
        )
    raise ValueError(f"Unsupported markov_head_type: {markov_head_type!r}")
