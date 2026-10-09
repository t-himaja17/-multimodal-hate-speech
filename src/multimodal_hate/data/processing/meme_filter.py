from dataclasses import dataclass
from typing import Optional


@dataclass
class MemeClassificationResult:
    """
    Result of meme-template classification.

    The methodology specifies using a ResNet-50 classifier
    fine-tuned on Know Your Meme data.
    """

    is_meme: bool
    confidence: float
    model_name: str = "ResNet-50"
    source: str = "Know Your Meme"


def create_meme_classification_result(
    is_meme: bool,
    confidence: float,
    model_name: str = "ResNet-50",
    source: str = "Know Your Meme",
) -> MemeClassificationResult:
    """
    Create a structured meme classification result.

    Parameters
    ----------
    is_meme:
        Whether the image is classified as a meme.

    confidence:
        Model confidence between 0.0 and 1.0.

    model_name:
        Name of the meme-template classifier.

    source:
        Dataset used for model fine-tuning.

    Returns
    -------
    MemeClassificationResult
        Structured classification result.
    """

    if not 0.0 <= confidence <= 1.0:
        raise ValueError(
            "Classification confidence must be between 0.0 and 1.0."
        )

    if not model_name or not model_name.strip():
        raise ValueError("Model name cannot be empty.")

    if not source or not source.strip():
        raise ValueError("Classification source cannot be empty.")

    return MemeClassificationResult(
        is_meme=is_meme,
        confidence=confidence,
        model_name=model_name.strip(),
        source=source.strip(),
    )


def is_meme_image(
    result: MemeClassificationResult,
    threshold: float = 0.5,
) -> bool:
    """
    Determine whether an image should be retained as a meme.

    Images classified as memes with confidence greater than
    or equal to the threshold are retained.
    """

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "Classification threshold must be between 0.0 and 1.0."
        )

    return result.is_meme and result.confidence >= threshold