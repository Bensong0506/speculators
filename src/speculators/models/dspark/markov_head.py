import torch
from torch import nn

__all__ = [
    "VanillaMarkovHead",
]


class VanillaMarkovHead(nn.Module):
    """Low-rank Markov correction head used by DSpark.

    For each draft position, the backbone emits base logits U_k. The Markov
    head looks at the previous token x_{k-1} and adds a low-rank bias
    W2(W1[x_{k-1}]) to U_k.
    """

    def __init__(
        self,
        *,
        verifier_vocab_size: int,
        draft_vocab_size: int,
        markov_rank: int,
    ) -> None:
        super().__init__()
        if markov_rank <= 0:
            raise ValueError(f"markov_rank must be positive, got {markov_rank}")
        self.verifier_vocab_size = verifier_vocab_size
        self.draft_vocab_size = draft_vocab_size
        self.markov_rank = markov_rank
        self.prev_token_embed = nn.Embedding(verifier_vocab_size, markov_rank)
        self.bias_proj = nn.Linear(markov_rank, draft_vocab_size, bias=False)

    def get_prev_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.prev_token_embed(token_ids.long())

    def project_bias(self, latent_states: torch.Tensor) -> torch.Tensor:
        return self.bias_proj(latent_states)

    def forward(self, prev_token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        prev_embeddings = self.get_prev_embeddings(prev_token_ids)
        return prev_embeddings, self.project_bias(prev_embeddings)
