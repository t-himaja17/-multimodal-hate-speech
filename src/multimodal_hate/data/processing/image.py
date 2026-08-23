from pathlib import Path
from typing import Optional

from PIL import Image


def validate_image(image_path: Optional[str]) -> bool:
    """
    Validate that the supplied path points to a readable image.

    The project methodology specifies that only image posts are retained.
    It does not specify image resizing, cropping, normalization, or other
    transformations at this stage.

    Therefore, this function performs validation only.
    """

    if not image_path:
        return False

    path = Path(image_path)

    if not path.is_file():
        return False

    try:
        with Image.open(path) as image:
            image.verify()
    except (OSError, ValueError):
        return False

    return True