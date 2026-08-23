"""
End-to-end sarcasm-aware multimodal hate-speech model.

Member 4 integration owner.

Pipeline:

    Image
      ↓
    ViT-L/14
      ↓
    Projection
      ↓
    Image tokens [B, N_img, 512]

    OCR / text
      ↓
    HateBERT
      ↓
    Token representations
      ↓
    Projection
      ↓
    Text tokens [B, N_txt, 512]

    CLIP image + CLIP text
      ↓
    Visual-text incongruity
      ↓
    [B, 1]

    SarcasmBERT
      ↓
    sarcasm probability [B, 1]

    Sentiment reversal
      ↓
    [B, 1]

    Three sarcasm signals
      ↓
    SarcasmGate
      ↓
    [B, gate_dim]

    Image + text tokens + sarcasm gate
      ↓
    SarcasmAwareFusionTransformer
      ↓
    pooled fused representation [B, 512]
      ↓
    MultitaskClassificationHeads
      ├── hate [B, 2]
      ├── sarcasm [B, 2]
      └── target [B, 5]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn

from multimodal_hate.models.encoders.vit import ViTEncoder
from multimodal_hate.models.encoders.hatebert import HateBERTEncoder
from multimodal_hate.models.encoders.clip import (
    CLIPImageEncoder,
    CLIPTextEncoder,
)

from multimodal_hate.models.fusion.projection import (
    ProjectionAlignment,
)

from multimodal_hate.models.fusion.fusion_transformer import (
    SarcasmAwareFusionTransformer,
)

from multimodal_hate.models.classification.heads import (
    MultitaskClassificationHeads,
)

from multimodal_hate.models.sarcasm.sarcasm_bert import (
    SarcasmBERT,
)

from multimodal_hate.models.sarcasm.incongruity import (
    VisualTextIncongruity,
)

from multimodal_hate.models.sarcasm.sentiment import (
    SentimentReversal,
)

from multimodal_hate.models.sarcasm.gate import (
    SarcasmGate,
)


@dataclass
class MultimodalHateSpeechOutput:
    """
    Complete output of the multimodal hate-speech model.

    Attributes
    ----------
    logits:
        Classification logits containing:
            hate   [B, 2]
            sarcasm [B, 2]
            target [B, 5]

    fused_tokens:
        Complete fused token representation
        [B, N_img + N_txt, 512].

    pooled:
        Final pooled multimodal representation [B, 512].

    image_representation:
        Mean-pooled projected image representation [B, 512].
        Used by the contrastive loss.

    text_representation:
        Mean-pooled projected text representation [B, 512].
        Used by the contrastive loss.

    sarcasm_gate:
        Learned sarcasm-conditioning vector [B, gate_dim].

    sarcasm_probability:
        SarcasmBERT probability [B, 1].

    incongruity_score:
        Visual-text incongruity score [B, 1].

    sentiment_reversal:
        Sentiment reversal indicator [B, 1].
    """

    logits: dict[str, Tensor]
    fused_tokens: Tensor
    pooled: Tensor
    image_representation: Tensor
    text_representation: Tensor
    sarcasm_gate: Tensor
    sarcasm_probability: Tensor
    incongruity_score: Tensor
    sentiment_reversal: Tensor


class MultimodalHateSpeechModel(nn.Module):
    """
    End-to-end sarcasm-aware multimodal hate-speech model.

    External encoders are injectable so that the model can be tested
    using lightweight dummy modules without loading large pretrained
    checkpoints.

    Default architecture:

        ViT-L/14             → 1024
        HateBERT             → 768
        CLIP image           → 512
        CLIP text            → 512
        shared fusion space  → 512
        sarcasm gate         → 64
        fusion layers        → 4
        attention heads      → 8

    Parameters
    ----------
    image_encoder:
        ViT-L/14 image encoder.

        Expected output:
            [B, N_img, 1024]

    text_encoder:
        HateBERT text encoder.

        Preferred interface:
            encode_tokens(texts) → [B, N_txt, 768]

        The encoder's normal forward() method returns only the CLS
        representation [B, 768], so encode_tokens() is required
        for transformer fusion.

    clip_image_encoder:
        CLIP image encoder.

        Expected output:
            [B, 512]

    clip_text_encoder:
        CLIP text encoder.

        Expected output:
            [B, 512]

    sarcasm_encoder:
        Optional SarcasmBERT module.

    sentiment_module:
        Optional SentimentReversal module.

    fusion_dim:
        Shared multimodal fusion dimension.

    sarcasm_gate_dim:
        Dimension of the sarcasm gate.

    num_fusion_layers:
        Number of cross-modal fusion layers.

    num_heads:
        Number of attention heads.

    dropout:
        Dropout probability.
    """

    def __init__(
        self,
        image_encoder: Optional[nn.Module] = None,
        text_encoder: Optional[nn.Module] = None,
        clip_image_encoder: Optional[nn.Module] = None,
        clip_text_encoder: Optional[nn.Module] = None,
        sarcasm_encoder: Optional[nn.Module] = None,
        sentiment_module: Optional[nn.Module] = None,
        fusion_dim: int = 512,
        sarcasm_gate_dim: int = 64,
        num_fusion_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if fusion_dim <= 0:
            raise ValueError(
                "fusion_dim must be greater than zero."
            )

        if sarcasm_gate_dim <= 0:
            raise ValueError(
                "sarcasm_gate_dim must be greater than zero."
            )

        if num_fusion_layers <= 0:
            raise ValueError(
                "num_fusion_layers must be greater than zero."
            )

        if num_heads <= 0:
            raise ValueError(
                "num_heads must be greater than zero."
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must be in the range [0, 1)."
            )

        self.fusion_dim = fusion_dim
        self.sarcasm_gate_dim = sarcasm_gate_dim

        # --------------------------------------------------------
        # Encoders
        # --------------------------------------------------------

        self.image_encoder = (
            image_encoder
            if image_encoder is not None
            else ViTEncoder()
        )

        self.text_encoder = (
            text_encoder
            if text_encoder is not None
            else HateBERTEncoder()
        )

        self.clip_image_encoder = (
            clip_image_encoder
            if clip_image_encoder is not None
            else CLIPImageEncoder()
        )

        self.clip_text_encoder = (
            clip_text_encoder
            if clip_text_encoder is not None
            else CLIPTextEncoder()
        )

        self.sarcasm_encoder = sarcasm_encoder
        self.sentiment_module = sentiment_module

        # --------------------------------------------------------
        # Determine encoder dimensions
        # --------------------------------------------------------

        image_input_dim = getattr(
            self.image_encoder,
            "output_dim",
            1024,
        )

        text_input_dim = getattr(
            self.text_encoder,
            "output_dim",
            768,
        )

        # --------------------------------------------------------
        # Projection
        #
        # ViT:
        #     [B, N_img, 1024]
        #          ↓
        #     [B, N_img, 512]
        #
        # HateBERT:
        #     [B, N_txt, 768]
        #          ↓
        #     [B, N_txt, 512]
        # --------------------------------------------------------

        self.projection = ProjectionAlignment(
            image_input_dim=image_input_dim,
            text_input_dim=text_input_dim,
            fusion_dim=fusion_dim,
        )

        # --------------------------------------------------------
        # Visual-text incongruity
        # --------------------------------------------------------

        self.incongruity = VisualTextIncongruity(
            embedding_dim=512,
        )

        # --------------------------------------------------------
        # Sarcasm gate
        # --------------------------------------------------------

        self.sarcasm_gate = SarcasmGate(
            gate_dim=sarcasm_gate_dim,
            dropout=dropout,
        )

        # --------------------------------------------------------
        # Cross-modal fusion transformer
        # --------------------------------------------------------

        self.fusion = SarcasmAwareFusionTransformer(
            fusion_dim=fusion_dim,
            num_layers=num_fusion_layers,
            num_heads=num_heads,
            dropout=dropout,
            sarcasm_gate_dim=sarcasm_gate_dim,
        )

        # --------------------------------------------------------
        # Classification heads
        # --------------------------------------------------------

        self.classification_heads = MultitaskClassificationHeads(
            input_dim=fusion_dim,
            dropout=dropout,
        )

    # ============================================================
    # VALIDATION
    # ============================================================

    @staticmethod
    def _validate_images(
        images: Tensor,
    ) -> None:
        """Validate image tensor."""

        if not isinstance(images, Tensor):
            raise TypeError(
                "images must be a torch.Tensor."
            )

        if images.ndim != 4:
            raise ValueError(
                "images must have shape [B, 3, H, W]. "
                f"Got {tuple(images.shape)}."
            )

        if images.size(1) != 3:
            raise ValueError(
                "images must have three channels. "
                f"Got {images.size(1)} channels."
            )

    @staticmethod
    def _validate_batch_size(
        batch_size: int,
        tensor: Tensor,
        name: str,
    ) -> None:
        """Ensure a tensor has the expected batch size."""

        if not isinstance(tensor, Tensor):
            raise TypeError(
                f"{name} must return a torch.Tensor."
            )

        if tensor.size(0) != batch_size:
            raise ValueError(
                f"{name} batch size does not match "
                f"image batch size. "
                f"Expected {batch_size}, got {tensor.size(0)}."
            )

    # ============================================================
    # TEXT TOKEN EXTRACTION
    # ============================================================

    def _encode_text_tokens(
        self,
        texts,
    ) -> Tensor:
        """
        Obtain token-level text representations.

        HateBERT now exposes:

            encode_tokens(texts)
                → [B, N_txt, 768]

        while its normal forward() method intentionally remains:

            forward(texts)
                → [B, 768]

        The fusion transformer requires token-level features,
        therefore encode_tokens() is preferred.

        A fallback to forward() is intentionally NOT converted into
        fake token sequences because that would silently produce an
        invalid multimodal representation.
        """

        encode_tokens = getattr(
            self.text_encoder,
            "encode_tokens",
            None,
        )

        if encode_tokens is None:
            raise AttributeError(
                "The text encoder must provide an "
                "encode_tokens(texts) method for multimodal "
                "transformer fusion."
            )

        text_tokens = encode_tokens(texts)

        if not isinstance(text_tokens, Tensor):
            raise TypeError(
                "text_encoder.encode_tokens() must return "
                "a torch.Tensor."
            )

        if text_tokens.ndim != 3:
            raise ValueError(
                "text_encoder.encode_tokens() must return "
                "[B, N_txt, D]. "
                f"Got {tuple(text_tokens.shape)}."
            )

        return text_tokens

    # ============================================================
    # SARCASM SIGNALS
    # ============================================================

    def _compute_sarcasm_probability(
        self,
        texts,
        sarcasm_probability: Optional[Tensor],
    ) -> Tensor:
        """
        Obtain sarcasm probability.

        If an explicit probability is supplied, it is used directly.

        Otherwise the configured SarcasmBERT module is required.

        Returns:
            [B, 1]
        """

        if sarcasm_probability is not None:

            if not isinstance(
                sarcasm_probability,
                Tensor,
            ):
                raise TypeError(
                    "sarcasm_probability must be a "
                    "torch.Tensor."
                )

            if sarcasm_probability.ndim != 2:
                raise ValueError(
                    "sarcasm_probability must have "
                    "shape [B, 1]."
                )

            if sarcasm_probability.size(1) != 1:
                raise ValueError(
                    "sarcasm_probability must have "
                    "shape [B, 1]."
                )

            return sarcasm_probability

        if self.sarcasm_encoder is None:
            raise ValueError(
                "sarcasm_encoder is required when "
                "sarcasm_probability is not supplied."
            )

        output = self.sarcasm_encoder(
            texts
        )

        if hasattr(
            output,
            "probability",
        ):
            probability = output.probability
        else:
            probability = output

        if not isinstance(
            probability,
            Tensor,
        ):
            raise TypeError(
                "Sarcasm encoder must return a tensor "
                "or an object containing a probability "
                "tensor."
            )

        if probability.ndim != 2:
            raise ValueError(
                "Sarcasm probability must have shape [B, 1]."
            )

        if probability.size(1) != 1:
            raise ValueError(
                "Sarcasm probability must have shape [B, 1]."
            )

        return probability

    def _compute_sentiment_reversal(
        self,
        texts,
        sentiment_reversal: Optional[Tensor],
    ) -> Tensor:
        """
        Obtain sentiment-reversal signal.

        If explicitly supplied, use it.

        Otherwise use the configured SentimentReversal module.

        Returns:
            [B, 1]
        """

        if sentiment_reversal is not None:

            reversal = sentiment_reversal

        else:

            if self.sentiment_module is None:
                raise ValueError(
                    "sentiment_module is required when "
                    "sentiment_reversal is not supplied."
                )

            reversal = self.sentiment_module(
                texts
            )

        if not isinstance(
            reversal,
            Tensor,
        ):
            raise TypeError(
                "sentiment_reversal must be a "
                "torch.Tensor."
            )

        if reversal.ndim != 2:
            raise ValueError(
                "sentiment_reversal must have "
                "shape [B, 1]."
            )

        if reversal.size(1) != 1:
            raise ValueError(
                "sentiment_reversal must have "
                "shape [B, 1]."
            )

        return reversal

    # ============================================================
    # FORWARD
    # ============================================================

    def forward(
        self,
        images: Tensor,
        texts,
        sarcasm_probability: Optional[Tensor] = None,
        sentiment_reversal: Optional[Tensor] = None,
    ) -> MultimodalHateSpeechOutput:
        """
        Run the complete multimodal model.

        Parameters
        ----------
        images:
            Image tensor [B, 3, H, W].

        texts:
            OCR/template text strings.

        sarcasm_probability:
            Optional externally supplied sarcasm probability
            [B, 1].

            If omitted, SarcasmBERT is used.

        sentiment_reversal:
            Optional externally supplied sentiment-reversal signal
            [B, 1].

            If omitted, SentimentReversal is used.

        Returns
        -------
        MultimodalHateSpeechOutput
            Complete model output.
        """

        # --------------------------------------------------------
        # Input validation
        # --------------------------------------------------------

        self._validate_images(
            images
        )

        batch_size = images.size(0)

        # ========================================================
        # 1. ViT image tokens
        # ========================================================

        image_tokens_raw = self.image_encoder(
            images
        )

        if image_tokens_raw.ndim != 3:
            raise ValueError(
                "Image encoder must return token features "
                "with shape [B, N_img, D]. "
                f"Got {tuple(image_tokens_raw.shape)}."
            )

        self._validate_batch_size(
            batch_size,
            image_tokens_raw,
            "image encoder",
        )

        # ========================================================
        # 2. HateBERT text tokens
        #
        # IMPORTANT:
        #
        # Do NOT call:
        #
        #     self.text_encoder(texts)
        #
        # because HateBERT.forward() returns [B, 768].
        #
        # Fusion needs:
        #
        #     [B, N_txt, 768]
        #
        # therefore use encode_tokens().
        # ========================================================

        text_tokens_raw = self._encode_text_tokens(
            texts
        )

        self._validate_batch_size(
            batch_size,
            text_tokens_raw,
            "text encoder",
        )

        # ========================================================
        # 3. Project image/text tokens into shared 512-D space
        # ========================================================

        image_tokens, text_tokens = self.projection(
            image_tokens_raw,
            text_tokens_raw,
        )

        if image_tokens.ndim != 3:
            raise ValueError(
                "Projected image tokens must have shape "
                "[B, N_img, fusion_dim]. "
                f"Got {tuple(image_tokens.shape)}."
            )

        if text_tokens.ndim != 3:
            raise ValueError(
                "Projected text tokens must have shape "
                "[B, N_txt, fusion_dim]. "
                f"Got {tuple(text_tokens.shape)}."
            )

        self._validate_batch_size(
            batch_size,
            image_tokens,
            "projected image tokens",
        )

        self._validate_batch_size(
            batch_size,
            text_tokens,
            "projected text tokens",
        )

        # ========================================================
        # 4. CLIP image representation
        # ========================================================

        clip_image = self.clip_image_encoder(
            images
        )

        if not isinstance(
            clip_image,
            Tensor,
        ):
            raise TypeError(
                "CLIP image encoder must return "
                "a torch.Tensor."
            )

        if clip_image.ndim != 2:
            raise ValueError(
                "CLIP image encoder must return "
                "[B, 512]. "
                f"Got {tuple(clip_image.shape)}."
            )

        if clip_image.size(1) != 512:
            raise ValueError(
                "CLIP image representation must have "
                "dimension 512. "
                f"Got {clip_image.size(1)}."
            )

        self._validate_batch_size(
            batch_size,
            clip_image,
            "CLIP image encoder",
        )

        # ========================================================
        # 5. CLIP text representation
        # ========================================================

        clip_text = self.clip_text_encoder(
            texts
        )

        if not isinstance(
            clip_text,
            Tensor,
        ):
            raise TypeError(
                "CLIP text encoder must return "
                "a torch.Tensor."
            )

        if clip_text.ndim != 2:
            raise ValueError(
                "CLIP text encoder must return "
                "[B, 512]. "
                f"Got {tuple(clip_text.shape)}."
            )

        if clip_text.size(1) != 512:
            raise ValueError(
                "CLIP text representation must have "
                "dimension 512. "
                f"Got {clip_text.size(1)}."
            )

        self._validate_batch_size(
            batch_size,
            clip_text,
            "CLIP text encoder",
        )

        # ========================================================
        # 6. Visual-text incongruity
        # ========================================================

        incongruity_score = self.incongruity(
            clip_image,
            clip_text,
        )

        if incongruity_score.ndim != 2:
            raise ValueError(
                "Incongruity module must return "
                "[B, 1]. "
                f"Got {tuple(incongruity_score.shape)}."
            )

        if incongruity_score.size(1) != 1:
            raise ValueError(
                "Incongruity module must return "
                "[B, 1]."
            )

        self._validate_batch_size(
            batch_size,
            incongruity_score,
            "incongruity score",
        )

        # ========================================================
        # 7. Sarcasm probability
        # ========================================================

        sarcasm_probability = (
            self._compute_sarcasm_probability(
                texts,
                sarcasm_probability,
            )
        )

        self._validate_batch_size(
            batch_size,
            sarcasm_probability,
            "sarcasm probability",
        )

        # ========================================================
        # 8. Sentiment reversal
        # ========================================================

        sentiment_reversal = (
            self._compute_sentiment_reversal(
                texts,
                sentiment_reversal,
            )
        )

        self._validate_batch_size(
            batch_size,
            sentiment_reversal,
            "sentiment reversal",
        )

        # ========================================================
        # 9. Sarcasm gate
        #
        # Inputs:
        #
        #     sarcasm probability
        #     incongruity
        #     sentiment reversal
        #
        # Output:
        #
        #     [B, gate_dim]
        # ========================================================

        sarcasm_gate = self.sarcasm_gate(
            sarcasm_probability,
            incongruity_score,
            sentiment_reversal,
        )

        if sarcasm_gate.ndim != 2:
            raise ValueError(
                "Sarcasm gate must return "
                "[B, gate_dim]. "
                f"Got {tuple(sarcasm_gate.shape)}."
            )

        if sarcasm_gate.size(1) != self.sarcasm_gate_dim:
            raise ValueError(
                "Unexpected sarcasm gate dimension. "
                f"Expected {self.sarcasm_gate_dim}, "
                f"got {sarcasm_gate.size(1)}."
            )

        self._validate_batch_size(
            batch_size,
            sarcasm_gate,
            "sarcasm gate",
        )

        # ========================================================
        # 10. Sarcasm-aware cross-modal fusion
        # ========================================================

        fused_tokens, pooled = self.fusion(
            image_tokens,
            text_tokens,
            sarcasm_gate,
        )

        if fused_tokens.ndim != 3:
            raise ValueError(
                "Fusion transformer must return fused tokens "
                "with shape [B, N_total, fusion_dim]. "
                f"Got {tuple(fused_tokens.shape)}."
            )

        if pooled.ndim != 2:
            raise ValueError(
                "Fusion transformer must return pooled "
                "representation [B, fusion_dim]. "
                f"Got {tuple(pooled.shape)}."
            )

        self._validate_batch_size(
            batch_size,
            fused_tokens,
            "fused tokens",
        )

        self._validate_batch_size(
            batch_size,
            pooled,
            "pooled representation",
        )

        if pooled.size(1) != self.fusion_dim:
            raise ValueError(
                "Unexpected pooled fusion dimension. "
                f"Expected {self.fusion_dim}, "
                f"got {pooled.size(1)}."
            )

        # ========================================================
        # 11. Multitask classification
        # ========================================================

        logits = self.classification_heads(
            pooled
        )

        # Validate expected classification outputs.

        if "hate" not in logits:
            raise ValueError(
                "Classification heads must return "
                "'hate' logits."
            )

        if "sarcasm" not in logits:
            raise ValueError(
                "Classification heads must return "
                "'sarcasm' logits."
            )

        if "target" not in logits:
            raise ValueError(
                "Classification heads must return "
                "'target' logits."
            )

        if logits["hate"].shape != (
            batch_size,
            2,
        ):
            raise ValueError(
                "Hate logits must have shape [B, 2]. "
                f"Got {tuple(logits['hate'].shape)}."
            )

        if logits["sarcasm"].shape != (
            batch_size,
            2,
        ):
            raise ValueError(
                "Sarcasm logits must have shape [B, 2]. "
                f"Got {tuple(logits['sarcasm'].shape)}."
            )

        if logits["target"].shape != (
            batch_size,
            5,
        ):
            raise ValueError(
                "Target logits must have shape [B, 5]. "
                f"Got {tuple(logits['target'].shape)}."
            )

        # ========================================================
        # 12. Contrastive representations
        #
        # Mean-pool projected image/text tokens.
        #
        # Image:
        #     [B, N_img, 512]
        #          ↓
        #     [B, 512]
        #
        # Text:
        #     [B, N_txt, 512]
        #          ↓
        #     [B, 512]
        #
        # These are consumed by NT-Xent.
        # ========================================================

        image_representation = image_tokens.mean(
            dim=1
        )

        text_representation = text_tokens.mean(
            dim=1
        )

        # ========================================================
        # 13. Return complete model output
        # ========================================================

        return MultimodalHateSpeechOutput(
            logits=logits,
            fused_tokens=fused_tokens,
            pooled=pooled,
            image_representation=image_representation,
            text_representation=text_representation,
            sarcasm_gate=sarcasm_gate,
            sarcasm_probability=sarcasm_probability,
            incongruity_score=incongruity_score,
            sentiment_reversal=sentiment_reversal,
        )