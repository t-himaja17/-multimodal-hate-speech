"""Sentiment reversal module for sarcasm detection.

Methodology:
    VADER sentiment
        +
    configurable BERT sentiment
        ↓
    polarity reversal
        ↓
    reversal indicator [B, 1]

The module supports region-aware text processing such as TOP and
BOTTOM regions when those regions are available.

No sentiment checkpoint is hard-coded because the research
methodology does not specify a particular BERT sentiment model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import torch
from torch import Tensor, nn

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "vaderSentiment is required for the sentiment module."
    ) from exc


@dataclass
class SentimentResult:
    """Container for sentiment-reversal outputs."""

    reversal: Tensor
    vader_polarity: Tensor
    bert_polarity: Tensor


class SentimentReversal(nn.Module):
    """Estimate polarity reversal using VADER and BERT sentiment.

    Args:
        tokenizer:
            BERT sentiment tokenizer.

        model:
            BERT sentiment model.

        device:
            Device used for BERT inference.

        max_length:
            Maximum number of tokens passed to BERT.

        reversal_threshold:
            Minimum disagreement between VADER and BERT polarity
            required to mark a polarity reversal.

        vader_threshold:
            Absolute VADER compound-score threshold used to determine
            positive/negative polarity.

    Expected input:
        texts:
            List of strings or a single string.

        Optional region texts:
            Mapping such as:
                {
                    "TOP": [...],
                    "BOTTOM": [...]
                }

    Output:
        reversal:
            Tensor with shape [B, 1], values in [0, 1].
    """

    def __init__(
        self,
        tokenizer: Any,
        model: nn.Module,
        device: str | torch.device = "cpu",
        max_length: int = 128,
        reversal_threshold: float = 0.5,
        vader_threshold: float = 0.05,
    ) -> None:
        super().__init__()

        if tokenizer is None:
            raise ValueError(
                "A BERT sentiment tokenizer is required."
            )

        if model is None:
            raise ValueError(
                "A BERT sentiment model is required."
            )

        if max_length <= 0:
            raise ValueError(
                "max_length must be greater than zero."
            )

        if reversal_threshold < 0.0:
            raise ValueError(
                "reversal_threshold must be non-negative."
            )

        if vader_threshold < 0.0:
            raise ValueError(
                "vader_threshold must be non-negative."
            )

        self.tokenizer = tokenizer
        self.model = model
        self.device = torch.device(device)
        self.max_length = max_length
        self.reversal_threshold = reversal_threshold
        self.vader_threshold = vader_threshold

        self.vader = SentimentIntensityAnalyzer()

        self.model.to(self.device)

    @staticmethod
    def _normalize_texts(
        texts: str | Sequence[str],
    ) -> list[str]:
        """Normalize a string or sequence of strings to a list."""

        if isinstance(texts, str):
            return [texts]

        if not isinstance(texts, Sequence):
            raise TypeError(
                "texts must be a string or a sequence of strings."
            )

        normalized = list(texts)

        if not normalized:
            raise ValueError(
                "texts cannot be empty."
            )

        if not all(
            isinstance(text, str)
            for text in normalized
        ):
            raise TypeError(
                "Every text item must be a string."
            )

        return normalized

    def vader_scores(
        self,
        texts: str | Sequence[str],
    ) -> Tensor:
        """Return VADER compound polarity scores.

        Output:
            [B, 1]
        """

        normalized = self._normalize_texts(texts)

        scores = [
            self.vader.polarity_scores(text)["compound"]
            for text in normalized
        ]

        return torch.tensor(
            scores,
            dtype=torch.float32,
        ).unsqueeze(1)

    def _vader_polarity(
        self,
        vader_scores: Tensor,
    ) -> Tensor:
        """Convert VADER compound scores to polarity.

        Returns:
            -1 for negative
             0 for neutral
            +1 for positive
        """

        polarity = torch.zeros_like(
            vader_scores
        )

        polarity = torch.where(
            vader_scores > self.vader_threshold,
            torch.ones_like(polarity),
            polarity,
        )

        polarity = torch.where(
            vader_scores < -self.vader_threshold,
            -torch.ones_like(polarity),
            polarity,
        )

        return polarity

    @staticmethod
    def _extract_logits(
        model_output: Any,
    ) -> Tensor:
        """Extract classification logits from common model outputs."""

        if hasattr(model_output, "logits"):
            logits = model_output.logits

        elif isinstance(model_output, Mapping):
            if "logits" not in model_output:
                raise ValueError(
                    "BERT model output does not contain 'logits'."
                )

            logits = model_output["logits"]

        elif isinstance(model_output, Tensor):
            logits = model_output

        else:
            raise TypeError(
                "Unsupported BERT model output type."
            )

        if not isinstance(logits, Tensor):
            raise TypeError(
                "BERT sentiment logits must be a torch.Tensor."
            )

        if logits.ndim != 2:
            raise ValueError(
                "BERT sentiment logits must have shape [B, C]."
            )

        return logits

    def bert_probabilities(
        self,
        texts: str | Sequence[str],
    ) -> Tensor:
        """Return BERT sentiment class probabilities.

        The expected BERT output is [B, C].

        For binary sentiment:
            class 0 = negative
            class 1 = positive

        For multi-class sentiment:
            the lowest-index class is treated as negative,
            the highest-index class as positive, and intermediate
            classes are treated as neutral.
        """

        normalized = self._normalize_texts(texts)

        encoded = self.tokenizer(
            normalized,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        if not isinstance(encoded, Mapping):
            raise TypeError(
                "BERT tokenizer must return a mapping."
            )

        encoded = {
            key: value.to(self.device)
            for key, value in encoded.items()
            if isinstance(value, Tensor)
        }

        if not encoded:
            raise ValueError(
                "Tokenizer returned no tensor inputs."
            )

        self.model.eval()

        with torch.no_grad():
            output = self.model(
                **encoded
            )

        logits = self._extract_logits(
            output
        )

        if logits.shape[0] != len(normalized):
            raise ValueError(
                "BERT model batch size does not match input batch."
            )

        return torch.softmax(
            logits,
            dim=-1,
        )

    def bert_polarity(
        self,
        texts: str | Sequence[str],
    ) -> Tensor:
        """Convert BERT sentiment probabilities to polarity.

        Returns:
            Tensor [B, 1] containing -1, 0, or +1.
        """

        probabilities = self.bert_probabilities(
            texts
        )

        class_count = probabilities.shape[1]

        if class_count < 2:
            raise ValueError(
                "BERT sentiment model must provide at least "
                "two sentiment classes."
            )

        if class_count == 2:
            negative_probability = probabilities[:, 0]
            positive_probability = probabilities[:, 1]

            polarity = torch.where(
                positive_probability > negative_probability,
                torch.ones_like(positive_probability),
                -torch.ones_like(positive_probability),
            )

            return polarity.unsqueeze(1)

        negative_probability = probabilities[:, 0]
        positive_probability = probabilities[:, -1]

        polarity = torch.zeros_like(
            positive_probability
        )

        polarity = torch.where(
            positive_probability > negative_probability,
            torch.ones_like(polarity),
            polarity,
        )

        polarity = torch.where(
            negative_probability > positive_probability,
            -torch.ones_like(polarity),
            polarity,
        )

        return polarity.unsqueeze(1)

    def _calculate_reversal(
        self,
        vader_polarity: Tensor,
        bert_polarity: Tensor,
    ) -> Tensor:
        """Calculate a binary polarity-reversal indicator."""

        if vader_polarity.shape != bert_polarity.shape:
            raise ValueError(
                "VADER and BERT polarity shapes must match."
            )

        disagreement = (
            vader_polarity
            * bert_polarity
        )

        reversal = (
            disagreement < 0
        ).to(torch.float32)

        return reversal

    def _combine_regions(
        self,
        region_texts: Mapping[str, Sequence[str]],
    ) -> list[str]:
        """Combine available text regions in deterministic order.

        Supported region names include TOP and BOTTOM.

        Missing regions are not fabricated.
        """

        if not isinstance(region_texts, Mapping):
            raise TypeError(
                "region_texts must be a mapping."
            )

        available_regions = []

        for region_name in ("TOP", "BOTTOM"):
            if region_name not in region_texts:
                continue

            region = region_texts[region_name]

            if isinstance(region, str):
                region = [region]

            if not isinstance(region, Sequence):
                raise TypeError(
                    f"Region {region_name} must contain strings."
                )

            region = list(region)

            if not all(
                isinstance(text, str)
                for text in region
            ):
                raise TypeError(
                    f"Region {region_name} must contain strings."
                )

            available_regions.append(
                region
            )

        if not available_regions:
            raise ValueError(
                "At least one supported region, TOP or BOTTOM, "
                "must be provided."
            )

        batch_sizes = {
            len(region)
            for region in available_regions
        }

        if len(batch_sizes) != 1:
            raise ValueError(
                "All provided regions must have the same batch size."
            )

        batch_size = next(
            iter(batch_sizes)
        )

        combined = []

        for index in range(batch_size):
            parts = [
                region[index]
                for region in available_regions
            ]

            combined.append(
                " ".join(
                    part
                    for part in parts
                    if part.strip()
                )
            )

        return combined

    def analyze(
        self,
        texts: str | Sequence[str] | None = None,
        region_texts: Optional[
            Mapping[str, Sequence[str]]
        ] = None,
    ) -> SentimentResult:
        """Analyze sentiment and calculate polarity reversal.

        Either ``texts`` or ``region_texts`` must be supplied.

        Region-aware processing combines available TOP and BOTTOM
        text for each sample.
        """

        if texts is not None and region_texts is not None:
            raise ValueError(
                "Provide either texts or region_texts, not both."
            )

        if region_texts is not None:
            normalized = self._combine_regions(
                region_texts
            )

        elif texts is not None:
            normalized = self._normalize_texts(
                texts
            )

        else:
            raise ValueError(
                "Either texts or region_texts must be provided."
            )

        vader_scores = self.vader_scores(
            normalized
        )

        vader_polarity = self._vader_polarity(
            vader_scores
        )

        bert_polarity = self.bert_polarity(
            normalized
        ).cpu()

        reversal = self._calculate_reversal(
            vader_polarity,
            bert_polarity,
        )

        return SentimentResult(
            reversal=reversal,
            vader_polarity=vader_polarity,
            bert_polarity=bert_polarity,
        )

    def forward(
        self,
        texts: str | Sequence[str] | None = None,
        region_texts: Optional[
            Mapping[str, Sequence[str]]
        ] = None,
    ) -> Tensor:
        """Return polarity-reversal indicator [B, 1]."""

        result = self.analyze(
            texts=texts,
            region_texts=region_texts,
        )

        return result.reversal