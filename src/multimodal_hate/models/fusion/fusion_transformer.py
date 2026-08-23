"""
Sarcasm-Aware Cross-Modal Transformer Fusion.

Member 4 ownership.

Architecture:
    Image tokens
        ↓
    Image → Text cross-attention
        ↓
    Text → Image cross-attention
        ↓
    [Image ; Text] fused self-attention
        ↓
    Repeat for 4 layers

Sarcasm-conditioned attention bias is supplied by
SarcasmConditionedAttentionBias.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from multimodal_hate.models.fusion.sarcasm_bias import (
    SarcasmConditionedAttentionBias,
)


class CrossModalFusionLayer(nn.Module):
    """
    One cross-modal fusion layer.

    The layer performs:

        1. Image → Text cross-attention
        2. Text → Image cross-attention
        3. Fused [IMG ; TXT] self-attention
        4. Feed-forward processing

    Parameters
    ----------
    fusion_dim:
        Shared multimodal representation dimension.

    num_heads:
        Number of attention heads.

    dropout:
        Dropout probability.

    sarcasm_gate_dim:
        Dimension of the sarcasm gate.
    """

    def __init__(
        self,
        fusion_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        sarcasm_gate_dim: int = 64,
    ) -> None:
        super().__init__()

        if fusion_dim <= 0:
            raise ValueError("fusion_dim must be greater than 0.")

        if num_heads <= 0:
            raise ValueError("num_heads must be greater than 0.")

        if fusion_dim % num_heads != 0:
            raise ValueError(
                "fusion_dim must be divisible by num_heads. "
                f"Got fusion_dim={fusion_dim}, "
                f"num_heads={num_heads}."
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must be in the range [0, 1)."
            )

        self.fusion_dim = fusion_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.sarcasm_gate_dim = sarcasm_gate_dim

        # Image queries attend to text keys/values.
        self.image_to_text = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Text queries attend to image keys/values.
        self.text_to_image = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Joint multimodal self-attention.
        self.fused_self_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.image_norm_1 = nn.LayerNorm(fusion_dim)
        self.text_norm_1 = nn.LayerNorm(fusion_dim)
        self.fused_norm = nn.LayerNorm(fusion_dim)

        self.image_norm_2 = nn.LayerNorm(fusion_dim)
        self.text_norm_2 = nn.LayerNorm(fusion_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 4, fusion_dim),
            nn.Dropout(dropout),
        )

        self.output_norm = nn.LayerNorm(fusion_dim)

        self.sarcasm_bias = SarcasmConditionedAttentionBias(
            gate_dim=sarcasm_gate_dim,
            num_heads=num_heads,
        )

    def _attention_with_sarcasm_bias(
        self,
        attention: nn.MultiheadAttention,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        sarcasm_gate: Tensor,
    ) -> Tensor:
        """
        Run multi-head attention and apply a sarcasm-conditioned
        additive bias to the attention logits.

        PyTorch's standard MultiheadAttention does not expose its
        attention logits directly. Therefore, the attention operation
        itself is performed here explicitly so the sarcasm bias can
        be inserted before softmax.
        """

        batch_size = query.size(0)

        query_len = query.size(1)
        key_len = key.size(1)

        q = attention.in_proj_weight[: attention.embed_dim]
        k = attention.in_proj_weight[
            attention.embed_dim : 2 * attention.embed_dim
        ]
        v = attention.in_proj_weight[
            2 * attention.embed_dim :
        ]

        if attention.in_proj_bias is not None:
            q_bias = attention.in_proj_bias[: attention.embed_dim]
            k_bias = attention.in_proj_bias[
                attention.embed_dim : 2 * attention.embed_dim
            ]
            v_bias = attention.in_proj_bias[
                2 * attention.embed_dim :
            ]
        else:
            q_bias = k_bias = v_bias = None

        query_projected = nn.functional.linear(
            query,
            q,
            q_bias,
        )

        key_projected = nn.functional.linear(
            key,
            k,
            k_bias,
        )

        value_projected = nn.functional.linear(
            value,
            v,
            v_bias,
        )

        head_dim = attention.embed_dim // attention.num_heads

        query_projected = query_projected.view(
            batch_size,
            query_len,
            attention.num_heads,
            head_dim,
        ).transpose(1, 2)

        key_projected = key_projected.view(
            batch_size,
            key_len,
            attention.num_heads,
            head_dim,
        ).transpose(1, 2)

        value_projected = value_projected.view(
            batch_size,
            key_len,
            attention.num_heads,
            head_dim,
        ).transpose(1, 2)

        scale = head_dim ** -0.5

        attention_logits = torch.matmul(
            query_projected,
            key_projected.transpose(-2, -1),
        ) * scale

        sarcasm_bias = self.sarcasm_bias(
            sarcasm_gate,
        )

        attention_logits = (
            attention_logits + sarcasm_bias
        )

        attention_weights = torch.softmax(
            attention_logits,
            dim=-1,
        )

        if attention.training and attention.dropout > 0:
            attention_weights = nn.functional.dropout(
                attention_weights,
                p=attention.dropout,
                training=True,
            )

        attended = torch.matmul(
            attention_weights,
            value_projected,
        )

        attended = attended.transpose(
            1,
            2,
        ).contiguous()

        attended = attended.view(
            batch_size,
            query_len,
            attention.embed_dim,
        )

        return nn.functional.linear(
            attended,
            attention.out_proj.weight,
            attention.out_proj.bias,
        )

    def forward(
        self,
        image_tokens: Tensor,
        text_tokens: Tensor,
        sarcasm_gate: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        image_tokens:
            [B, N_img, fusion_dim]

        text_tokens:
            [B, N_txt, fusion_dim]

        sarcasm_gate:
            [B, sarcasm_gate_dim]

        Returns
        -------
        image_tokens:
            [B, N_img, fusion_dim]

        text_tokens:
            [B, N_txt, fusion_dim]
        """

        self._validate_inputs(
            image_tokens,
            text_tokens,
            sarcasm_gate,
        )

        # --------------------------------------------------------
        # Image → Text
        # --------------------------------------------------------

        image_context = self._attention_with_sarcasm_bias(
            self.image_to_text,
            query=image_tokens,
            key=text_tokens,
            value=text_tokens,
            sarcasm_gate=sarcasm_gate,
        )

        image_tokens = self.image_norm_1(
            image_tokens + image_context
        )

        # --------------------------------------------------------
        # Text → Image
        # --------------------------------------------------------

        text_context = self._attention_with_sarcasm_bias(
            self.text_to_image,
            query=text_tokens,
            key=image_tokens,
            value=image_tokens,
            sarcasm_gate=sarcasm_gate,
        )

        text_tokens = self.text_norm_1(
            text_tokens + text_context
        )

        # --------------------------------------------------------
        # Fused self-attention
        # --------------------------------------------------------

        fused_tokens = torch.cat(
            [
                image_tokens,
                text_tokens,
            ],
            dim=1,
        )

        fused_context = self._attention_with_sarcasm_bias(
            self.fused_self_attention,
            query=fused_tokens,
            key=fused_tokens,
            value=fused_tokens,
            sarcasm_gate=sarcasm_gate,
        )

        fused_tokens = self.fused_norm(
            fused_tokens + fused_context
        )

        # --------------------------------------------------------
        # Feed-forward network
        # --------------------------------------------------------

        fused_tokens = self.output_norm(
            fused_tokens + self.feed_forward(
                fused_tokens
            )
        )

        image_length = image_tokens.size(1)

        image_tokens = fused_tokens[
            :, :image_length, :
        ]

        text_tokens = fused_tokens[
            :, image_length:, :
        ]

        return image_tokens, text_tokens

    def _validate_inputs(
        self,
        image_tokens: Tensor,
        text_tokens: Tensor,
        sarcasm_gate: Tensor,
    ) -> None:

        if image_tokens.ndim != 3:
            raise ValueError(
                "image_tokens must have shape [B, N_img, D]. "
                f"Got {tuple(image_tokens.shape)}."
            )

        if text_tokens.ndim != 3:
            raise ValueError(
                "text_tokens must have shape [B, N_txt, D]. "
                f"Got {tuple(text_tokens.shape)}."
            )

        if sarcasm_gate.ndim != 2:
            raise ValueError(
                "sarcasm_gate must have shape [B, d]. "
                f"Got {tuple(sarcasm_gate.shape)}."
            )

        if image_tokens.size(0) != text_tokens.size(0):
            raise ValueError(
                "Image and text batch sizes must match."
            )

        if image_tokens.size(0) != sarcasm_gate.size(0):
            raise ValueError(
                "Image tokens and sarcasm gate batch sizes "
                "must match."
            )

        if image_tokens.size(-1) != self.fusion_dim:
            raise ValueError(
                f"Expected image fusion dimension "
                f"{self.fusion_dim}, got "
                f"{image_tokens.size(-1)}."
            )

        if text_tokens.size(-1) != self.fusion_dim:
            raise ValueError(
                f"Expected text fusion dimension "
                f"{self.fusion_dim}, got "
                f"{text_tokens.size(-1)}."
            )


class SarcasmAwareFusionTransformer(nn.Module):
    """
    Four-layer sarcasm-aware multimodal transformer.

    Default methodology configuration:

        fusion dimension = 512
        layers           = 4
        attention heads  = 8
        dropout          = 0.1

    Input:

        image_tokens  [B, N_img, 512]
        text_tokens   [B, N_txt, 512]
        sarcasm_gate  [B, d]

    Output:

        fused_tokens   [B, N_img + N_txt, 512]
        pooled         [B, 512]
    """

    def __init__(
        self,
        fusion_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        sarcasm_gate_dim: int = 64,
    ) -> None:
        super().__init__()

        if num_layers <= 0:
            raise ValueError(
                "num_layers must be greater than 0."
            )

        self.fusion_dim = fusion_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.sarcasm_gate_dim = sarcasm_gate_dim

        self.layers = nn.ModuleList(
            [
                CrossModalFusionLayer(
                    fusion_dim=fusion_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    sarcasm_gate_dim=sarcasm_gate_dim,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(fusion_dim)

    def forward(
        self,
        image_tokens: Tensor,
        text_tokens: Tensor,
        sarcasm_gate: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Run the complete multimodal fusion transformer.

        Returns
        -------
        fused_tokens:
            [B, N_img + N_txt, fusion_dim]

        pooled:
            [B, fusion_dim]
        """

        for layer in self.layers:
            image_tokens, text_tokens = layer(
                image_tokens,
                text_tokens,
                sarcasm_gate,
            )

        fused_tokens = torch.cat(
            [
                image_tokens,
                text_tokens,
            ],
            dim=1,
        )

        fused_tokens = self.final_norm(
            fused_tokens
        )

        pooled = fused_tokens.mean(dim=1)

        return fused_tokens, pooled