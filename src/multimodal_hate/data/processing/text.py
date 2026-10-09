from typing import Optional


def preprocess_text(text: Optional[str]) -> str:
    """
    Prepare text for the multimodal hate-speech pipeline.

    The project methodology does not specify a particular text
    normalization strategy such as lowercasing, stop-word removal,
    stemming, or lemmatization.

    Therefore, this function only performs safe whitespace handling
    while preserving the original textual content.

    Unspecified normalization/tokenization steps remain TBD.
    """

    if text is None:
        return ""

    return " ".join(text.split())


def combine_ocr_and_context(
    ocr_text: Optional[str],
    post_context: Optional[str],
) -> str:
    """
    Combine OCR text with post-title/context information.

    The project methodology specifies that OCR text is merged with
    post-title information as structured context.

    Positional OCR tags such as [TOP], [BOTTOM], and [CAPTION]
    are preserved.
    """

    ocr = preprocess_text(ocr_text)
    context = preprocess_text(post_context)

    parts = []

    if ocr:
        parts.append(ocr)

    if context:
        parts.append(context)

    return " ".join(parts)