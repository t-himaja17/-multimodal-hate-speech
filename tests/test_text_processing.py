from src.multimodal_hate.data.processing.text import (
    preprocess_text,
    combine_ocr_and_context,
)


def test_preprocess_text_preserves_content():
    text = "  This is   a meme.  "

    result = preprocess_text(text)

    assert result == "This is a meme."


def test_preprocess_text_handles_none():
    assert preprocess_text(None) == ""


def test_combine_ocr_and_context():
    ocr_text = "[TOP] Example text [BOTTOM] Another text"
    post_context = "Post title context"

    result = combine_ocr_and_context(
        ocr_text,
        post_context,
    )

    assert result == (
        "[TOP] Example text [BOTTOM] Another text "
        "Post title context"
    )


def test_combine_ocr_and_context_with_missing_ocr():
    result = combine_ocr_and_context(
        None,
        "Post title context",
    )

    assert result == "Post title context"