from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image


@dataclass
class OCRResult:
    """
    Structured representation of OCR output for a single text region.
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


def _assign_region(
    bounding_box: Tuple[int, int, int, int],
    image_height: int,
) -> str:
    """
    Assign a simple positional region to an OCR bounding box.

    TOP:
        Upper third of the image.

    BOTTOM:
        Lower third of the image.

    CAPTION:
        Middle region.
    """

    _, y1, _, y2 = bounding_box

    center_y = (y1 + y2) / 2.0

    if image_height <= 0:
        return "CAPTION"

    relative_y = center_y / image_height

    if relative_y < 0.33:
        return "TOP"

    if relative_y > 0.67:
        return "BOTTOM"

    return "CAPTION"


def extract_text_easyocr(
    image_path: str,
    gpu: bool = True,
    min_confidence: float = 0.15,
) -> List[OCRResult]:
    """
    Extract text from an image using EasyOCR.

    Returns structured OCR results containing:
        - text
        - bounding box
        - positional region
        - confidence
    """

    import easyocr

    reader = easyocr.Reader(
        ["en"],
        gpu=gpu,
    )

    image = Image.open(image_path).convert("RGB")
    image_array = np.asarray(image)

    image_height = image_array.shape[0]

    raw_results = reader.readtext(image_array)

    results: List[OCRResult] = []

    for item in raw_results:
        if len(item) != 3:
            continue

        coordinates, text, confidence = item

        if not text or not text.strip():
            continue

        confidence = float(confidence)

        if confidence < min_confidence:
            continue

        xs = [int(point[0]) for point in coordinates]
        ys = [int(point[1]) for point in coordinates]

        bounding_box = (
            min(xs),
            min(ys),
            max(xs),
            max(ys),
        )

        region = _assign_region(
            bounding_box,
            image_height,
        )

        results.append(
            create_ocr_result(
                text=text,
                bounding_box=bounding_box,
                region=region,
                confidence=confidence,
            )
        )

    results.sort(
        key=lambda result: (
            result.bounding_box[1],
            result.bounding_box[0],
        )
    )

    return results


def extract_text_from_image(
    image_path: str,
    gpu: bool = True,
) -> str:
    """
    Extract and format OCR text from an image.

    This is the main image-only OCR entry point used by inference.

    The output contains positional OCR tags:
        [TOP]
        [CAPTION]
        [BOTTOM]
    """

    results = extract_text_easyocr(
        image_path=image_path,
        gpu=gpu,
    )

    return format_ocr_text(results)