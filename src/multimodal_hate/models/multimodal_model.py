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
      └── target [B, 4]
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

from multimodal_hate.models.sarcasm.incongruity import (
    VisualTextIncongruity,
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
        Classification logits:

            hate     [B, 2]
            sarcasm  [B, 2]
            target   [B, 4]

    fused_tokens:
        Complete fused token representation:

            [B, N_img + N_txt, fusion_dim]

    pooled:
        Final pooled multimodal representation:

            [B, fusion_dim]

    image_representation:
        Mean-pooled projected image representation:

            [B, fusion_dim]

        Used by NT-Xent contrastive loss.

    text_representation:
        Mean-pooled projected text representation:

            [B, fusion_dim]

        Used by NT-Xent contrastive loss.

    sarcasm_gate:
        Learned sarcasm-conditioning vector:

            [B, gate_dim]

    sarcasm_probability:
        SarcasmBERT probability:

            [B, 1]

    incongruity_score:
        Visual-text incongruity score:

            [B, 1]

    sentiment_reversal:
        Sentiment reversal indicator:

            [B, 1]
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

    Default architecture:

        ViT-L/14             -> 1024
        HateBERT             -> 768
        CLIP image           -> 512
        CLIP text            -> 512
        shared fusion space  -> 512
        sarcasm gate         -> 64
        fusion layers        -> 4
        attention heads      -> 8

    External encoders are injectable so that the complete model can
    be tested with lightweight dummy modules without downloading
    pretrained checkpoints.

    Important text interface:

        HateBERT.forward(texts)
            -> [B, 768]

        HateBERT.encode_tokens(texts)
            -> [B, N_txt, 768]

    The fusion transformer requires token-level representations,
    therefore encode_tokens() is mandatory for text fusion.
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
        self.sarcasm_gate_dim = sarcasm_gate_dim

        # ========================================================
        # ENCODERS
        # ========================================================

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

        # ========================================================
        # ENCODER DIMENSIONS
        # ========================================================

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

        # ========================================================
        # PROJECTION / MODALITY ALIGNMENT
        # ========================================================

        self.projection = ProjectionAlignment(
            image_input_dim=image_input_dim,
            text_input_dim=text_input_dim,
            fusion_dim=fusion_dim,
        )

        # ========================================================
        # VISUAL-TEXT INCONGRUITY
        # ========================================================

        self.incongruity = VisualTextIncongruity(
            embedding_dim=512,
        )

        # ========================================================
        # SARCASM GATE
        # ========================================================

        self.sarcasm_gate = SarcasmGate(
            gate_dim=sarcasm_gate_dim,
            dropout=dropout,
        )

        # ========================================================
        # CROSS-MODAL FUSION TRANSFORMER
        # ========================================================

        self.fusion = SarcasmAwareFusionTransformer(
            fusion_dim=fusion_dim,
            num_layers=num_fusion_layers,
            num_heads=num_heads,
            dropout=dropout,
            sarcasm_gate_dim=sarcasm_gate_dim,
        )

        # ========================================================
        # MULTITASK CLASSIFICATION HEADS
        # ========================================================

        self.classification_heads = MultitaskClassificationHeads(
            input_dim=fusion_dim,
            dropout=dropout,
        )

    # ============================================================
    # VALIDATION HELPERS
    # ============================================================

    @staticmethod
    def _validate_images(
        images: Tensor,
    ) -> None:
        """Validate input image tensor."""

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

        if images.size(0) <= 0:
            raise ValueError(
                "Image batch must contain at least one sample."
            )

    @staticmethod
    def _validate_batch_size(
        batch_size: int,
        tensor: Tensor,
        name: str,
    ) -> None:
        """Validate tensor type and batch dimension."""

        if not isinstance(tensor, Tensor):
            raise TypeError(
                f"{name} must return a torch.Tensor."
            )

        if tensor.ndim == 0:
            raise ValueError(
                f"{name} must have a batch dimension."
            )

        if tensor.size(0) != batch_size:
            raise ValueError(
                f"{name} batch size does not match image "
                f"batch size. Expected {batch_size}, "
                f"got {tensor.size(0)}."
            )

    @staticmethod
    def _validate_signal(
        signal: Tensor,
        batch_size: int,
        name: str,
    ) -> None:
        """Validate a [B, 1] scalar signal."""

        if not isinstance(signal, Tensor):
            raise TypeError(
                f"{name} must be a torch.Tensor."
            )

        if signal.ndim != 2 or signal.size(1) != 1:
            raise ValueError(
                f"{name} must have shape [B, 1]. "
                f"Got {tuple(signal.shape)}."
            )

        if signal.size(0) != batch_size:
            raise ValueError(
                f"{name} batch size does not match image "
                f"batch size. Expected {batch_size}, "
                f"got {signal.size(0)}."
            )

    # ============================================================
    # TEXT TOKEN EXTRACTION
    # ============================================================

    def _encode_text_tokens(
        self,
        texts,
    ) -> Tensor:
        """
        Obtain token-level HateBERT representations.

        Required interface:

            encode_tokens(texts)
                -> [B, N_txt, 768]

        We intentionally do not fall back to forward() because
        forward() returns only the CLS representation [B, 768].
        """

        encode_tokens = getattr(
            self.text_encoder,
            "encode_tokens",
            None,
        )

        if encode_tokens is None:
            raise AttributeError(
                "The text encoder must provide "
                "encode_tokens(texts) for multimodal "
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

        if text_tokens.size(1) <= 0:
            raise ValueError(
                "Text encoder returned zero tokens."
            )

        return text_tokens

    # ============================================================
    # SARCASM PROBABILITY
    # ============================================================

    def _compute_sarcasm_probability(
        self,
        texts,
        sarcasm_probability: Optional[Tensor],
        batch_size: int,
        reference_tensor: Tensor,
    ) -> Tensor:
        """
        Obtain sarcasm probability.

        If supplied externally, use it.

        Otherwise use the configured SarcasmBERT module.

        Returned tensor is moved to the same device and dtype as
        the reference tensor so it can safely enter SarcasmGate.
        """

        if sarcasm_probability is not None:

            probability = sarcasm_probability

        else:

            if self.sarcasm_encoder is None:
                raise ValueError(
                    "sarcasm_encoder is required when "
                    "sarcasm_probability is not supplied."
                )

            output = self.sarcasm_encoder(texts)

            if hasattr(
                output,
                "probability",
            ):
                probability = output.probability
            else:
                probability = output

        self._validate_signal(
            probability,
            batch_size,
            "sarcasm_probability",
        )

        probability = probability.to(
            device=reference_tensor.device,
            dtype=reference_tensor.dtype,
        )

        return probability

    # ============================================================
    # SENTIMENT REVERSAL
    # ============================================================

    def _compute_sentiment_reversal(
        self,
        texts,
        sentiment_reversal: Optional[Tensor],
        batch_size: int,
        reference_tensor: Tensor,
    ) -> Tensor:
        """
        Obtain sentiment-reversal signal.

        If supplied externally, use it.

        Otherwise use the configured SentimentReversal module.

        Returned tensor is moved to the same device and dtype as
        the reference tensor.
        """

        if sentiment_reversal is not None:

            reversal = sentiment_reversal

        else:

            if self.sentiment_module is None:
                raise ValueError(
                    "sentiment_module is required when "
                    "sentiment_reversal is not supplied."
                )

            reversal = self.sentiment_module(texts)

        self._validate_signal(
            reversal,
            batch_size,
            "sentiment_reversal",
        )

        reversal = reversal.to(
            device=reference_tensor.device,
            dtype=reference_tensor.dtype,
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
        Run the complete sarcasm-aware multimodal model.

        Parameters
        ----------
        images:
            Image tensor:

                [B, 3, H, W]

        texts:
            OCR/template text strings.

        sarcasm_probability:
            Optional externally supplied sarcasm probability:

                [B, 1]

            If omitted, SarcasmBERT is used.

        sentiment_reversal:
            Optional externally supplied sentiment-reversal signal:

                [B, 1]

            If omitted, SentimentReversal is used.

        Returns
        -------
        MultimodalHateSpeechOutput
            Complete multimodal model output.
        """

        # ========================================================
        # INPUT VALIDATION
        # ========================================================

        self._validate_images(images)

        batch_size = images.size(0)

        # ========================================================
        # 1. ViT IMAGE TOKENS
        #
        # [B, N_img, 1024]
        # ========================================================

        image_tokens_raw = self.image_encoder(
            images
        )

        if not isinstance(
            image_tokens_raw,
            Tensor,
        ):
            raise TypeError(
                "Image encoder must return a torch.Tensor."
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
        # 2. HATEBERT TOKEN REPRESENTATIONS
        #
        # [B, N_txt, 768]
        #
        # IMPORTANT:
        #
        # Do NOT call:
        #
        #     self.text_encoder(texts)
        #
        # because that returns:
        #
        #     [B, 768]
        #
        # Fusion requires:
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
        # 3. PROJECTION / MODALITY ALIGNMENT
        #
        # Image:
        #
        #     [B, N_img, 1024]
        #             ↓
        #     [B, N_img, 512]
        #
        # Text:
        #
        #     [B, N_txt, 768]
        #             ↓
        #     [B, N_txt, 512]
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

        if image_tokens.size(-1) != self.fusion_dim:
            raise ValueError(
                "Projected image token dimension does not "
                f"match fusion_dim={self.fusion_dim}. "
                f"Got {image_tokens.size(-1)}."
            )

        if text_tokens.size(-1) != self.fusion_dim:
            raise ValueError(
                "Projected text token dimension does not "
                f"match fusion_dim={self.fusion_dim}. "
                f"Got {text_tokens.size(-1)}."
            )

        # ========================================================
        # 4. CLIP IMAGE REPRESENTATION
        #
        # [B, 512]
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
        # 5. CLIP TEXT REPRESENTATION
        #
        # [B, 512]
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
        # 6. VISUAL-TEXT INCONGRUITY
        #
        # CLIP image + CLIP text
        #       ↓
        # semantic dissimilarity
        #       ↓
        # learned incongruity score
        #
        # [B, 1]
        # ========================================================

        incongruity_score = self.incongruity(
            clip_image,
            clip_text,
        )

        self._validate_signal(
            incongruity_score,
            batch_size,
            "incongruity_score",
        )

        incongruity_score = incongruity_score.to(
            device=image_tokens.device,
            dtype=image_tokens.dtype,
        )

        # ========================================================
        # 7. SARCASM PROBABILITY
        #
        # [B, 1]
        # ========================================================

        sarcasm_probability = (
            self._compute_sarcasm_probability(
                texts=texts,
                sarcasm_probability=sarcasm_probability,
                batch_size=batch_size,
                reference_tensor=image_tokens,
            )
        )

        # ========================================================
        # 8. SENTIMENT REVERSAL
        #
        # [B, 1]
        # ========================================================

        sentiment_reversal = (
            self._compute_sentiment_reversal(
                texts=texts,
                sentiment_reversal=sentiment_reversal,
                batch_size=batch_size,
                reference_tensor=image_tokens,
            )
        )

        # ========================================================
        # 9. SARCASM GATE
        #
        # Three signals:
        #
        #   sarcasm probability
        #   incongruity score
        #   sentiment reversal
        #
        #             ↓
        #
        #   SarcasmGate
        #
        #             ↓
        #
        #   [B, gate_dim]
        # ========================================================

        sarcasm_gate = self.sarcasm_gate(
            sarcasm_probability,
            incongruity_score,
            sentiment_reversal,
        )

        if not isinstance(
            sarcasm_gate,
            Tensor,
        ):
            raise TypeError(
                "SarcasmGate must return a torch.Tensor."
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
        # 10. SARCASM-AWARE CROSS-MODAL FUSION
        #
        # Image tokens
        # Text tokens
        # Sarcasm gate
        #
        #        ↓
        #
        # 4-layer transformer
        #
        #        ↓
        #
        # fused_tokens:
        # [B, N_img + N_txt, 512]
        #
        # pooled:
        # [B, 512]
        # ========================================================

        fused_tokens, pooled = self.fusion(
            image_tokens,
            text_tokens,
            sarcasm_gate,
        )

        if not isinstance(
            fused_tokens,
            Tensor,
        ):
            raise TypeError(
                "Fusion transformer must return "
                "fused tokens as a torch.Tensor."
            )

        if not isinstance(
            pooled,
            Tensor,
        ):
            raise TypeError(
                "Fusion transformer must return "
                "pooled representation as a torch.Tensor."
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
        # 11. MULTITASK CLASSIFICATION
        #
        # pooled [B, 512]
        #
        #             ↓
        #
        # ┌─────────────────────────────┐
        # │ Hate       → [B, 2]         │
        # │ Sarcasm    → [B, 2]         │
        # │ Target     → [B, 4]         │
        # └─────────────────────────────┘
        # ========================================================

        logits = self.classification_heads(
            pooled
        )

        if not isinstance(
            logits,
            dict,
        ):
            raise TypeError(
                "Classification heads must return "
                "a dictionary of logits."
            )

        required_heads = (
            "hate",
            "sarcasm",
            "target",
        )

        for head_name in required_heads:

            if head_name not in logits:
                raise ValueError(
                    "Classification heads must return "
                    f"'{head_name}' logits."
                )

            if not isinstance(
                logits[head_name],
                Tensor,
            ):
                raise TypeError(
                    f"'{head_name}' logits must be "
                    "a torch.Tensor."
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
            4,
        ):
            raise ValueError(
                "Target logits must have shape [B, 4]. "
                f"Got {tuple(logits['target'].shape)}."
            )

        # ========================================================
        # 12. CONTRASTIVE REPRESENTATIONS
        #
        # Projected image tokens:
        #
        #     [B, N_img, 512]
        #             ↓ mean
        #     [B, 512]
        #
        # Projected text tokens:
        #
        #     [B, N_txt, 512]
        #             ↓ mean
        #     [B, 512]
        #
        # Used by NT-Xent contrastive loss.
        # ========================================================

        image_representation = image_tokens.mean(
            dim=1
        )

        text_representation = text_tokens.mean(
            dim=1
        )

        if image_representation.ndim != 2:
            raise ValueError(
                "Image contrastive representation must "
                "have shape [B, fusion_dim]."
            )

        if text_representation.ndim != 2:
            raise ValueError(
                "Text contrastive representation must "
                "have shape [B, fusion_dim]."
            )

        if image_representation.shape != (
            batch_size,
            self.fusion_dim,
        ):
            raise ValueError(
                "Unexpected image contrastive representation "
                f"shape: {tuple(image_representation.shape)}."
            )

        if text_representation.shape != (
            batch_size,
            self.fusion_dim,
        ):
            raise ValueError(
                "Unexpected text contrastive representation "
                f"shape: {tuple(text_representation.shape)}."
            )

        # ========================================================
        # 13. RETURN COMPLETE OUTPUT
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