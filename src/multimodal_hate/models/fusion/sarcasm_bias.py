"""
Sarcasm-conditioned attention bias.

Member 4 ownership.

The methodology specifies:

    attention_logits' =
        attention_logits + sarcasm_bias(g)

where g is the sarcasm gate produced by Member 3.

Important:
    The sarcasm gate dimension is configurable/TBD.
    This module therefore does NOT assume that d == 512.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SarcasmConditionedAttentionBias(nn.Module):
    """
    Converts the sarcasm gate into an additive attention bias.

    The gate dimension is configurable and is independent of the
    fusion dimension.

    Parameters
    ----------
    gate_dim:
        Dimension d of the sarcasm gate [B, d].

    num_heads:
        Number of attention heads.

    bias_scale:
        Optional initial scaling factor for the generated bias.

    Notes
    -----
    The exact broadcasting/projection mechanism is marked TBD by
    the project interface. This implementation uses a learnable
    projection from the sarcasm gate to one scalar per attention
    head:

        [B, d]
            ↓
        Linear
            ↓
        [B, num_heads]
            ↓
        [B, num_heads, 1, 1]

    This allows the bias to be broadcast across query and key
    positions while keeping the gate dimension independent from
    the fusion dimension.
    """

    def __init__(
        self,
        gate_dim: int,
        num_heads: int = 8,
        bias_scale: float = 1.0,
    ) -> None:
        super().__init__()

        if gate_dim <= 0:
            raise ValueError("gate_dim must be greater than 0.")

        if num_heads <= 0:
            raise ValueError("num_heads must be greater than 0.")

        if bias_scale < 0:
            raise ValueError("bias_scale must be non-negative.")

        self.gate_dim = gate_dim
        self.num_heads = num_heads
        self.bias_scale = bias_scale

        self.projection = nn.Linear(gate_dim, num_heads)

    def forward(self, sarcasm_gate: Tensor) -> Tensor:
        """
        Convert sarcasm gate into an additive attention bias.

        Parameters
        ----------
        sarcasm_gate:
            Tensor with shape [B, gate_dim].

        Returns
        -------
        Tensor:
            Attention bias with shape [B, num_heads, 1, 1].

        This can be broadcast against attention logits of shape:

            [B, num_heads, query_len, key_len]
        """

        if sarcasm_gate.ndim != 2:
            raise ValueError(
                "sarcasm_gate must have shape [B, d]. "
                f"Got {tuple(sarcasm_gate.shape)}."
            )

        if sarcasm_gate.size(-1) != self.gate_dim:
            raise ValueError(
                f"Expected sarcasm gate dimension {self.gate_dim}, "
                f"got {sarcasm_gate.size(-1)}."
            )

        bias = self.projection(sarcasm_gate)

        bias = bias * self.bias_scale

        return bias.unsqueeze(-1).unsqueeze(-1)

    def apply(
        self,
        attention_logits: Tensor,
        sarcasm_gate: Tensor,
    ) -> Tensor:
        """
        Add sarcasm-conditioned bias to attention logits.

        Parameters
        ----------
        attention_logits:
            Tensor of shape:

                [B, num_heads, query_len, key_len]

        sarcasm_gate:
            Tensor of shape:

                [B, gate_dim]

        Returns
        -------
        Tensor:
            Sarcasm-conditioned attention logits with the same
            shape as attention_logits.
        """

        if attention_logits.ndim != 4:
            raise ValueError(
                "attention_logits must have shape "
                "[B, num_heads, query_len, key_len]. "
                f"Got {tuple(attention_logits.shape)}."
            )

        if attention_logits.size(1) != self.num_heads:
            raise ValueError(
                f"Expected {self.num_heads} attention heads, "
                f"got {attention_logits.size(1)}."
            )

        if attention_logits.size(0) != sarcasm_gate.size(0):
            raise ValueError(
                "Attention logits and sarcasm gate must have "
                "the same batch size. "
                f"Got {attention_logits.size(0)} and "
                f"{sarcasm_gate.size(0)}."
            )

        bias = self.forward(sarcasm_gate)

        return attention_logits + bias