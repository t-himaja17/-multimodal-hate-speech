"""Auxiliary sarcasm prediction head.

This module provides a binary sarcasm prediction head compatible with
BCEWithLogitsLoss.

The methodology specifies:
    BCE loss
    lambda_1 = 0.3

The loss weight is intentionally NOT implemented here. Member 4
will integrate the auxiliary loss during training.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SarcasmAuxiliaryHead(nn.Module):
    """Binary sarcasm prediction head.

    Args:
        input_dim:
            Dimension of the incoming sarcasm representation.

        hidden_dim:
            Optional hidden dimension. If None, a direct linear
            classifier is used.

        dropout:
            Dropout probability when a hidden layer is used.

    Input:
        Tensor [B, input_dim]

    Output:
        Raw logits [B, 1]

    Training:
        Use torch.nn.BCEWithLogitsLoss with the returned logits.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if not isinstance(input_dim, int):
            raise TypeError("input_dim must be an integer.")

        if input_dim <= 0:
            raise ValueError(
                "input_dim must be greater than zero."
            )

        if hidden_dim is not None:
            if not isinstance(hidden_dim, int):
                raise TypeError(
                    "hidden_dim must be an integer or None."
                )

            if hidden_dim <= 0:
                raise ValueError(
                    "hidden_dim must be greater than zero."
                )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must satisfy 0 <= dropout < 1."
            )

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        if hidden_dim is None:
            self.classifier = nn.Linear(
                input_dim,
                1,
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

    def forward(self, x: Tensor) -> Tensor:
        """Return raw sarcasm logits with shape [B, 1]."""

        if not isinstance(x, Tensor):
            raise TypeError(
                "x must be a torch.Tensor."
            )

        if x.ndim != 2:
            raise ValueError(
                f"x must have shape [B, input_dim], "
                f"got {tuple(x.shape)}."
            )

        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected input dimension {self.input_dim}, "
                f"got {x.shape[1]}."
            )

        if x.shape[0] == 0:
            raise ValueError(
                "Batch dimension cannot be zero."
            )

        return self.classifier(x)

    @staticmethod
    def probability(logits: Tensor) -> Tensor:
        """Convert raw logits to sarcasm probabilities."""

        if not isinstance(logits, Tensor):
            raise TypeError(
                "logits must be a torch.Tensor."
            )

        if logits.ndim != 2 or logits.shape[1] != 1:
            raise ValueError(
                "logits must have shape [B, 1]."
            )

        return torch.sigmoid(logits)