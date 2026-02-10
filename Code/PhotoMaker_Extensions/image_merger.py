# image_merger.py
# Merge two generated images (left identity, right identity) into a single output

import numpy as np
from PIL import Image


def merge_images(
    left_image: Image.Image,
    right_image: Image.Image,
    **kwargs
) -> Image.Image:
    """
    Horizontally concatenate two images: full left image + full right image.
    Final width = left_width + right_width

    Args:
        left_image: PIL Image with left identity
        right_image: PIL Image with right identity

    Returns:
        Combined PIL Image (width = left + right)
    """
    # Ensure same height
    if left_image.height != right_image.height:
        target_height = max(left_image.height, right_image.height)
        left_image = left_image.resize(
            (int(left_image.width * target_height / left_image.height), target_height),
            Image.LANCZOS
        )
        right_image = right_image.resize(
            (int(right_image.width * target_height / right_image.height), target_height),
            Image.LANCZOS
        )

    # Concatenate: left + right
    total_width = left_image.width + right_image.width
    merged = Image.new('RGB', (total_width, left_image.height))
    merged.paste(left_image, (0, 0))
    merged.paste(right_image, (left_image.width, 0))

    return merged


def side_by_side(
    left_image: Image.Image,
    right_image: Image.Image,
    gap: int = 0,
) -> Image.Image:
    """
    Same as merge_images but with optional gap between images.
    """
    if left_image.height != right_image.height:
        target_height = max(left_image.height, right_image.height)
        left_image = left_image.resize(
            (int(left_image.width * target_height / left_image.height), target_height),
            Image.LANCZOS
        )
        right_image = right_image.resize(
            (int(right_image.width * target_height / right_image.height), target_height),
            Image.LANCZOS
        )

    total_width = left_image.width + right_image.width + gap
    combined = Image.new('RGB', (total_width, left_image.height), (255, 255, 255))
    combined.paste(left_image, (0, 0))
    combined.paste(right_image, (left_image.width + gap, 0))
    return combined
