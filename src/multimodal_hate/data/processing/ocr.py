from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps


@dataclass
class OCRResult:
    """Structured OCR output for a single text region."""

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
    if not text or not text.strip():
        raise ValueError("OCR text cannot be empty.")

    if len(bounding_box) != 4:
        raise ValueError("Bounding box must contain four coordinates.")

    if confidence is not None and not 0.0 <= confidence <= 1.0:
        raise ValueError("Confidence must be between 0 and 1.")

    return OCRResult(
        text=text.strip(),
        bounding_box=bounding_box,
        region=region,
        confidence=confidence,
    )


def format_ocr_text(results: List[OCRResult]) -> str:
    """Format recognized text with positional tags."""

    parts = []

    for result in results:
        if not result.text.strip():
            continue

        if result.region:
            parts.append(f"[{result.region.upper()}] {result.text}")
        else:
            parts.append(result.text)

    return " ".join(parts)


def _assign_region(
    bounding_box: Tuple[int, int, int, int],
    image_height: int,
) -> str:
    _, y1, _, y2 = bounding_box
    relative_y = ((y1 + y2) / 2.0) / max(1, image_height)

    if relative_y < 0.33:
        return "TOP"

    if relative_y > 0.67:
        return "BOTTOM"

    return "CAPTION"


@lru_cache(maxsize=1)
def _get_reader():
    """Load and cache RapidOCR for local inference."""

    from rapidocr import RapidOCR

    return RapidOCR()


def extract_text_easyocr(
    image_path: str,
    gpu: bool = True,
    min_confidence: float = 0.10,
) -> List[OCRResult]:
    """
    Preserve the existing function name for compatibility.
    Recognition is performed using RapidOCR and ONNX Runtime.
    The gpu argument is retained for compatibility.
    """

    reader = _get_reader()

    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image_array = np.asarray(image)

    output = reader(image_array)

    if output is None or output.boxes is None:
        return []

    image_height, image_width = image_array.shape[:2]
    results: List[OCRResult] = []

    for box_points, text, confidence in zip(
        output.boxes,
        output.txts,
        output.scores,
    ):
        text = " ".join(str(text).split())
        confidence = float(confidence)

        if not text or confidence < min_confidence:
            continue

        points = np.asarray(box_points, dtype=float)

        x1 = max(0, int(points[:, 0].min()))
        y1 = max(0, int(points[:, 1].min()))
        x2 = min(image_width, int(points[:, 0].max()))
        y2 = min(image_height, int(points[:, 1].max()))

        bounding_box = (x1, y1, x2, y2)

        results.append(
            create_ocr_result(
                text=text,
                bounding_box=bounding_box,
                region=_assign_region(
                    bounding_box,
                    image_height,
                ),
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
    """Public entry point used by the existing application."""

    results = extract_text_easyocr(
        image_path=image_path,
        gpu=gpu,
    )

    return format_ocr_text(results)