"""Training configuration model — the structured config produced by the Planner."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TrainingConfig(BaseModel):
    """Training recipe specified by the Planner or frozen from Phase 1 seed.

    The Code Executor generates this alongside the nn.Module code.
    The harness interprets it — the LLM never writes the training loop.
    """

    optimizer: str = "adamw"  # sgd, adam, adamw
    lr: float = 1e-3
    weight_decay: float = 0.01
    momentum: float = 0.9  # SGD only
    betas: tuple[float, float] = (0.9, 0.999)  # Adam/AdamW

    scheduler: str = "cosine"  # cosine, step, warmup_cosine, plateau, none
    scheduler_params: dict[str, Any] = Field(default_factory=dict)
    warmup_epochs: int = 0

    batch_size: int = 32
    epochs: int = 50

    augmentations: list[str] = Field(default_factory=list)  # mixup, cutmix, random_erasing
    augmentation_params: dict[str, Any] = Field(default_factory=dict)

    grad_clip: float | None = None
    label_smoothing: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrainingConfig":
        return cls(**d)
