# image_merger.py
# Merge two generated images (left identity, right identity) into a single output

import numpy as np
from PIL import Image


def merge_images(
    left_image: Image.Image,
    right_image: Image.Image,
    **kwargs  # Ignore any extra params for compatibility
) -> Image.Image:
    """
    Simple horizontal merge: left half from left_image + right half from right_image.
    No blending, just a clean cut at the center.

    Args:
        left_image: PIL Image with left identity
        right_image: PIL Image with right identity

    Returns:
        Merged PIL Image
    """
    # Ensure same size
    if left_image.size != right_image.size:
        right_image = right_image.resize(left_image.size, Image.LANCZOS)

    width, height = left_image.size
    center = width // 2

    # Convert to numpy
    left_arr = np.array(left_image)
    right_arr = np.array(right_image)

    # Simple merge: left half + right half
    merged = np.zeros_like(left_arr)
    merged[:, :center] = left_arr[:, :center]  # Left half from left image
    merged[:, center:] = right_arr[:, center:]  # Right half from right image

    return Image.fromarray(merged)


def side_by_side(
    left_image: Image.Image,
    right_image: Image.Image,
    gap: int = 0,
) -> Image.Image:
    """
    Side-by-side concatenation (doubles width).
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
