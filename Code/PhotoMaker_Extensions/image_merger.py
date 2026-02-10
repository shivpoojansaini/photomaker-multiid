# image_merger.py
# Merge two generated images (left identity, right identity) into a single output

import random
import numpy as np
from PIL import Image
from typing import Tuple, Optional


def merge_images(
    left_image: Image.Image,
    right_image: Image.Image,
    overlap_percent: Optional[float] = None,
    blend_mode: str = "smooth",
) -> Image.Image:
    """
    Merge two images horizontally: left half from left_image, right half from right_image.

    The merge takes:
    - Left portion (0 to ~50%) from left_image
    - Right portion (~50% to 100%) from right_image
    - Small overlap zone (5-20%) in the middle for natural blending

    Args:
        left_image: PIL Image with left identity
        right_image: PIL Image with right identity
        overlap_percent: Overlap percentage (5-20). If None, random between 5-20%
        blend_mode: "smooth" (sigmoid), "linear", or "hard" (no blend)

    Returns:
        Merged PIL Image with same dimensions as input
    """
    # Ensure same size
    if left_image.size != right_image.size:
        right_image = right_image.resize(left_image.size, Image.LANCZOS)

    width, height = left_image.size

    # Random overlap between 5-20% if not specified
    if overlap_percent is None:
        overlap_percent = random.uniform(5, 20)

    overlap_percent = max(5, min(20, overlap_percent))  # Clamp to 5-20%

    # Calculate overlap zone
    overlap_width = int(width * overlap_percent / 100)
    center = width // 2

    # Overlap zone boundaries
    blend_start = center - overlap_width // 2
    blend_end = center + overlap_width // 2

    # Convert to numpy
    left_arr = np.array(left_image).astype(np.float32)
    right_arr = np.array(right_image).astype(np.float32)

    # Create output array
    merged = np.zeros_like(left_arr)

    # Left region: 100% from left image
    merged[:, :blend_start] = left_arr[:, :blend_start]

    # Right region: 100% from right image
    merged[:, blend_end:] = right_arr[:, blend_end:]

    # Blend region in the middle
    if blend_end > blend_start:
        blend_width = blend_end - blend_start

        if blend_mode == "hard":
            # Hard cut at center
            cut_point = blend_width // 2
            merged[:, blend_start:blend_start + cut_point] = left_arr[:, blend_start:blend_start + cut_point]
            merged[:, blend_start + cut_point:blend_end] = right_arr[:, blend_start + cut_point:blend_end]

        elif blend_mode == "linear":
            # Linear gradient blend
            alpha = np.linspace(1.0, 0.0, blend_width)[np.newaxis, :, np.newaxis]
            left_blend = left_arr[:, blend_start:blend_end]
            right_blend = right_arr[:, blend_start:blend_end]
            merged[:, blend_start:blend_end] = left_blend * alpha + right_blend * (1 - alpha)

        else:  # smooth (sigmoid)
            # Smooth sigmoid transition for natural look
            x = np.linspace(-4, 4, blend_width)  # sigmoid range
            alpha = 1.0 / (1.0 + np.exp(x))  # sigmoid: 1->0
            alpha = alpha[np.newaxis, :, np.newaxis]

            left_blend = left_arr[:, blend_start:blend_end]
            right_blend = right_arr[:, blend_start:blend_end]
            merged[:, blend_start:blend_end] = left_blend * alpha + right_blend * (1 - alpha)

    # Convert back to PIL
    merged = np.clip(merged, 0, 255).astype(np.uint8)
    return Image.fromarray(merged)


def side_by_side(
    left_image: Image.Image,
    right_image: Image.Image,
    gap: int = 0,
) -> Image.Image:
    """
    Simple side-by-side concatenation (no blending, doubles width).

    Args:
        left_image: Left PIL Image
        right_image: Right PIL Image
        gap: Gap in pixels between images

    Returns:
        Combined PIL Image (width = left + right + gap)
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

    # Create combined image
    total_width = left_image.width + right_image.width + gap
    combined = Image.new('RGB', (total_width, left_image.height), (255, 255, 255))

    combined.paste(left_image, (0, 0))
    combined.paste(right_image, (left_image.width + gap, 0))

    return combined
