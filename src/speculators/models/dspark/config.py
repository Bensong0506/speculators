from typing import Literal

from pydantic import Field

from speculators import SpeculatorModelConfig
from speculators.models.dflash.config import DFlashSpeculatorConfig

__all__ = [
    "DSparkSpeculatorConfig",
]


@SpeculatorModelConfig.register("dspark")
class DSparkSpeculatorConfig(DFlashSpeculatorConfig):
    """Configuration for DSpark-style DFlash training.

    DSpark keeps DFlash's anchor-block drafter, then adds a Markov correction
    head and a confidence head trained with DeepSeek's CE + L1 + confidence
    objective.
    """

    speculators_model_type: Literal["dspark"] = "dspark"
    architectures: list[str] = Field(
        default_factory=lambda: ["DSparkSpeculator"],
        description="Model architectures that can load these weights",
    )

    markov_rank: int = Field(
        default=256,
        description="Low-rank dimension for the Markov correction head.",
    )

    confidence_head_alpha: float = Field(
        default=1.0,
        description="Loss weight for the confidence head BCE objective.",
    )

    confidence_head_with_markov: bool = Field(
        default=True,
        description="Concatenate Markov previous-token embedding into confidence head.",
    )

    ce_loss_alpha: float = Field(
        default=0.1,
        description="Loss weight for hard-label cross entropy.",
    )

    l1_loss_alpha: float = Field(
        default=0.9,
        description="Loss weight for draft/target distribution L1 matching.",
    )

    loss_decay_gamma: float = Field(
        default=4.0,
        description="Exponential decay gamma for later positions in each block.",
    )
