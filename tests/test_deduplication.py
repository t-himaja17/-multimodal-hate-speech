import pytest
from PIL import Image

from src.multimodal_hate.data.processing.deduplication import (
    PHASH_SIMILARITY_THRESHOLD,
    compute_phash,
    deduplicate_image_hashes,
    is_duplicate,
    phash_similarity,
)


def test_compute_phash(tmp_path):
    image_path = tmp_path / "image.png"

    image = Image.new("RGB", (100, 100), "white")
    image.save(image_path)

    result = compute_phash(str(image_path))

    assert isinstance(result, str)
    assert len(result) > 0


def test_compute_phash_missing_image():
    with pytest.raises(FileNotFoundError):
        compute_phash("does_not_exist.png")


def test_identical_hashes_have_full_similarity():
    hash_value = "0000000000000000"

    similarity = phash_similarity(hash_value, hash_value)

    assert similarity == 1.0


def test_identical_hashes_are_duplicates():
    hash_value = "0000000000000000"

    assert is_duplicate(
        hash_value,
        hash_value,
        PHASH_SIMILARITY_THRESHOLD,
    )


def test_different_hashes_have_lower_similarity():
    hash_a = "0000000000000000"
    hash_b = "ffffffffffffffff"

    similarity = phash_similarity(hash_a, hash_b)

    assert similarity == 0.0


def test_invalid_hash_lengths():
    with pytest.raises(ValueError):
        phash_similarity(
            "0000000000000000",
            "0000",
        )


def test_invalid_threshold():
    with pytest.raises(ValueError):
        is_duplicate(
            "0000000000000000",
            "0000000000000000",
            threshold=1.5,
        )


def test_deduplicate_image_hashes():
    image_hashes = {
        "image_001": "0000000000000000",
        "image_002": "0000000000000000",
        "image_003": "ffffffffffffffff",
    }

    retained = deduplicate_image_hashes(
        image_hashes,
        threshold=PHASH_SIMILARITY_THRESHOLD,
    )

    assert retained == ["image_001", "image_003"]