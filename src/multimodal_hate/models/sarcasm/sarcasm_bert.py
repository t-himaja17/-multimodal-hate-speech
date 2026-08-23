"""SarcasmBERT interface for sarcasm-aware multimodal hate detection.

The methodology specifies:
    OCR text + template caption/context
        -> SarcasmBERT
        -> sarcasm probability + text representation

The exact pretrained SarcasmBERT checkpoint is not specified by the
project methodology, so the checkpoint is configurable rather than
hard-coded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch
from torch import Tensor, nn
from transformers import AutoModel, AutoTokenizer


@dataclass
class SarcasmBERTOutput:
    """Outputs returned by SarcasmBERT."""

    probability: Tensor
    representation: Tensor
    span_markers: Optional[Any] = None


class SarcasmBERT(nn.Module):
    """Configurable SarcasmBERT text encoder and sarcasm classifier.

    Args:
        model_name:
            Hugging Face model/checkpoint identifier.
            The project methodology does not currently specify
            the exact checkpoint, so this remains configurable.

        device:
            Device for computation. If omitted, CUDA is used when
            available; otherwise CPU.

        max_length:
            Maximum token sequence length.

        dropout:
            Dropout probability before the classification head.

        pretrained_model:
            Optional already-loaded Transformer model. Mainly useful
            for testing and custom integration.

        tokenizer:
            Optional already-loaded tokenizer. Mainly useful for
            testing and custom integration.

        representation_dim:
            Optional Transformer hidden dimension. If omitted, it is
            inferred from the model configuration.
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[torch.device | str] = None,
        max_length: int = 128,
        dropout: float = 0.1,
        pretrained_model: Optional[nn.Module] = None,
        tokenizer: Optional[Any] = None,
        representation_dim: Optional[int] = None,
    ) -> None:
        super().__init__()

        if not model_name and pretrained_model is None:
            raise ValueError(
                "model_name must be provided when pretrained_model "
                "is not supplied."
            )

        if max_length <= 0:
            raise ValueError("max_length must be greater than zero.")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1).")

        self.model_name = model_name
        self.max_length = max_length

        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        # Load tokenizer only when one was not supplied.
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            self.tokenizer = tokenizer

        # Load Transformer only when one was not supplied.
        if pretrained_model is None:
            self.transformer = AutoModel.from_pretrained(model_name)
        else:
            self.transformer = pretrained_model

        if representation_dim is None:
            representation_dim = self._infer_hidden_size(
                self.transformer
            )

        if representation_dim <= 0:
            raise ValueError(
                "representation_dim must be greater than zero."
            )

        self.representation_dim = representation_dim

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(
            representation_dim,
            1,
        )

        self.to(self.device)

    @staticmethod
    def _infer_hidden_size(model: nn.Module) -> int:
        """Infer the Transformer representation dimension."""

        config = getattr(model, "config", None)

        if config is None:
            raise ValueError(
                "Transformer model does not expose a configuration."
            )

        hidden_size = getattr(
            config,
            "hidden_size",
            None,
        )

        if hidden_size is None:
            hidden_size = getattr(
                config,
                "dim",
                None,
            )

        if hidden_size is None:
            raise ValueError(
                "Unable to infer Transformer hidden size. "
                "Pass representation_dim explicitly."
            )

        return int(hidden_size)

    @staticmethod
    def _validate_texts(
        texts: str | Sequence[str],
    ) -> list[str]:
        """Validate and normalize text input."""

        if isinstance(texts, str):
            return [texts]

        if not isinstance(texts, Sequence):
            raise TypeError(
                "texts must be a string or a sequence of strings."
            )

        normalized = list(texts)

        if not normalized:
            raise ValueError(
                "texts must contain at least one item."
            )

        if not all(
            isinstance(text, str)
            for text in normalized
        ):
            raise TypeError(
                "Every item in texts must be a string."
            )

        return normalized

    def _prepare_inputs(
        self,
        texts: list[str],
    ) -> dict[str, Tensor]:
        """Tokenize a batch of text inputs."""

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        if not isinstance(encoded, dict):
            encoded = dict(encoded)

        tensor_inputs: dict[str, Tensor] = {}

        for key, value in encoded.items():
            if isinstance(value, Tensor):
                tensor_inputs[key] = value.to(self.device)

        if "input_ids" not in tensor_inputs:
            raise ValueError(
                "Tokenizer output must contain 'input_ids'."
            )

        return tensor_inputs

    @staticmethod
    def _extract_representation(
        outputs: Any,
    ) -> Tensor:
        """Extract one representation vector per input."""

        last_hidden_state = getattr(
            outputs,
            "last_hidden_state",
            None,
        )

        if last_hidden_state is not None:
            if last_hidden_state.ndim != 3:
                raise ValueError(
                    "Expected last_hidden_state with shape [B, T, H]."
                )

            # First-token representation.
            return last_hidden_state[:, 0, :]

        pooler_output = getattr(
            outputs,
            "pooler_output",
            None,
        )

        if pooler_output is not None:
            if pooler_output.ndim != 2:
                raise ValueError(
                    "Expected pooler_output with shape [B, H]."
                )

            return pooler_output

        if isinstance(outputs, (tuple, list)) and outputs:
            first_output = outputs[0]

            if isinstance(first_output, Tensor):
                if first_output.ndim == 3:
                    return first_output[:, 0, :]

        raise ValueError(
            "Unable to extract a sequence representation from "
            "Transformer output."
        )

    def forward(
        self,
        texts: str | Sequence[str],
    ) -> SarcasmBERTOutput:
        """Run batched sarcasm inference.

        Args:
            texts:
                One text string or a sequence of text strings.

        Returns:
            SarcasmBERTOutput containing:

            probability:
                Sigmoid sarcasm probability with shape [B, 1].

            representation:
                Transformer representation with shape [B, H].

            span_markers:
                None unless the underlying model explicitly provides
                span information.
        """

        normalized_texts = self._validate_texts(texts)

        inputs = self._prepare_inputs(
            normalized_texts
        )

        outputs = self.transformer(
            **inputs
        )

        representation = self._extract_representation(
            outputs
        )

        if representation.ndim != 2:
            raise ValueError(
                "Sarcasm representation must have shape [B, H]."
            )

        if representation.shape[0] != len(
            normalized_texts
        ):
            raise ValueError(
                "Representation batch dimension does not "
                "match input batch."
            )

        representation = self.dropout(
            representation
        )

        logits = self.classifier(
            representation
        )

        # Apply sigmoid exactly once.
        probability = torch.sigmoid(
            logits
        )

        return SarcasmBERTOutput(
            probability=probability,
            representation=representation,
            span_markers=None,
        )

    def predict_probability(
        self,
        texts: str | Sequence[str],
    ) -> Tensor:
        """Return only the sarcasm probability."""

        return self.forward(
            texts
        ).probability