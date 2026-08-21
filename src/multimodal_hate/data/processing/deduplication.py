from pathlib import Path
from typing import Dict, List

import imagehash
from PIL import Image

PHASH_SIMILARITY_THRESHOLD = 0.95


def compute_phash(image_path: str) -> str:
    """
    Compute a perceptual hash for an image.
    """

    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(path) as image:
        return str(imagehash.phash(image.convert("RGB")))


def phash_similarity(hash_a: str, hash_b: str) -> float:
    """
    Calculate similarity between two perceptual hashes.

    Similarity is defined as:

        1 - (Hamming distance / hash length)

    Returns
    -------
    float
        Similarity in the range [0.0, 1.0].
    """

    if not hash_a or not hash_b:
        raise ValueError("Perceptual hashes cannot be empty.")

    if len(hash_a) != len(hash_b):
        raise ValueError("Perceptual hashes must have the same length.")

    try:
        value_a = int(hash_a, 16)
        value_b = int(hash_b, 16)
    except ValueError as exc:
        raise ValueError("Invalid hexadecimal perceptual hash.") from exc

    xor_value = value_a ^ value_b
    hamming_distance = xor_value.bit_count()
    hash_bits = len(hash_a) * 4

    return 1.0 - (hamming_distance / hash_bits)


def is_duplicate(
    hash_a: str,
    hash_b: str,
    threshold: float = PHASH_SIMILARITY_THRESHOLD,
) -> bool:
    """
    Determine whether two images are duplicates according to pHash similarity.

    The methodology specifies a similarity threshold of >= 95%.
    """

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be between 0.0 and 1.0.")

    return phash_similarity(hash_a, hash_b) >= threshold


def deduplicate_image_hashes(
    image_hashes: Dict[str, str],
    threshold: float = PHASH_SIMILARITY_THRESHOLD,
) -> List[str]:
    """
    Remove near-duplicate images using perceptual-hash similarity.

    Parameters
    ----------
    image_hashes:
        Mapping from image/sample ID to perceptual hash.

    threshold:
        Similarity threshold. The methodology specifies 0.95.

    Returns
    -------
    List[str]
        IDs retained after deduplication.

    Notes
    -----
    The first occurrence of a visually similar image is retained.
    """

    retained_ids: List[str] = []
    retained_hashes: List[str] = []

    for sample_id, image_hash in image_hashes.items():
        duplicate = any(
            is_duplicate(image_hash, existing_hash, threshold)
            for existing_hash in retained_hashes
        )

        if not duplicate:
            retained_ids.append(sample_id)
            retained_hashes.append(image_hash)

    return retained_ids