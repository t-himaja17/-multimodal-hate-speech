from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class OCRResult:
    """
    Structured representation of OCR output for a single text region.

    The methodology specifies that OCR processing should preserve:
    - extracted text
    - bounding boxes
    - region information
    - confidence scores
    """

    text: str
    bounding_box: Tuple[int, int, int, int]
    region: Optional[str] = None
    confidence: Optional[float] = None


def create_ocr_result(
    text: str,
    bounding_box: Tuple[int, int, int, int],
    region: Optional[str] = None,
    confidence: Optional[float] = None,
) -> OCRResult:
    """
    Create a structured OCR result.

    Parameters
    ----------
    text:
        OCR-extracted text.

    bounding_box:
        Bounding box represented as:
        (x1, y1, x2, y2)

    region:
        Positional region label such as:
        [TOP], [BOTTOM], or [CAPTION]

    confidence:
        OCR confidence score.

    Returns
    -------
    OCRResult
        Structured OCR result.
    """

    if not text or not text.strip():
        raise ValueError("OCR text cannot be empty.")

    if len(bounding_box) != 4:
        raise ValueError(
            "Bounding box must contain exactly four coordinates."
        )

    if confidence is not None and not 0.0 <= confidence <= 1.0:
        raise ValueError(
            "OCR confidence must be between 0.0 and 1.0."
        )

    return OCRResult(
        text=text.strip(),
        bounding_box=bounding_box,
        region=region,
        confidence=confidence,
    )


def format_ocr_text(results: List[OCRResult]) -> str:
    """
    Convert structured OCR results into positional OCR text.

    Supported positional tags are:

        [TOP]
        [BOTTOM]
        [CAPTION]

    Results without a region label are included without
    a positional tag.
    """

    if not results:
        return ""

    parts = []

    for result in results:
        text = result.text.strip()

        if not text:
            continue

        if result.region:
            parts.append(f"[{result.region.upper()}] {text}")
        else:
            parts.append(text)

    return " ".join(parts)