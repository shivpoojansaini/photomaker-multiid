# image_merger.py
# Merge two generated images (left identity, right identity) into a single output

import random
import numpy as np
from PIL import Image
from typing import Optional


def merge_images(
    left_image: Image.Image,
    right_image: Image.Image,
    overlap_percent: Optional[float] = None,
    blend_mode: str = "smooth",
) -> Image.Image:
    """
    Merge two images: left side from left_image + right side from right_image.

    Takes:
    - LEFT side of left_image (0 to ~50%)
    - RIGHT side of right_image (~50% to 100%)
    - Small 5-15% blend zone at the seam for natural transition

    Args:
        left_image: PIL Image with left identity
        right_image: PIL Image with right identity
        overlap_percent: Blend zone width as percentage (5-15). If None, random 5-15%
        blend_mode: "smooth" (sigmoid), "linear", or "hard" (no blend)

    Returns:
        Merged PIL Image with same dimensions as input
    """
    # Ensure same size
    if left_image.size != right_image.size:
        right_image = right_image.resize(left_image.size, Image.LANCZOS)

    width, height = left_image.size

    # Random overlap between 5-15% if not specified
    if overlap_percent is None:
        overlap_percent = random.uniform(5, 15)

    overlap_percent = max(5, min(15, overlap_percent))  # Clamp to 5-15%

    # Calculate blend zone (centered at 50%)
    blend_width = int(width * overlap_percent / 100)
    center = width // 2

    # Blend zone boundaries
    blend_start = center - blend_width // 2
    blend_end = center + blend_width // 2

    # Convert to numpy
    left_arr = np.array(left_image).astype(np.float32)
    right_arr = np.array(right_image).astype(np.float32)

    # Create output array
    merged = np.zeros_like(left_arr)

    # LEFT region: take from LEFT image (0 to blend_start)
    merged[:, :blend_start] = left_arr[:, :blend_start]

    # RIGHT region: take from RIGHT image (blend_end to width)
    merged[:, blend_end:] = right_arr[:, blend_end:]

    # Blend zone in the middle
    if blend_end > blend_start:
        bw = blend_end - blend_start

        if blend_mode == "hard":
            # Hard cut at center (no blending)
            cut = bw // 2
            merged[:, blend_start:blend_start + cut] = left_arr[:, blend_start:blend_start + cut]
            merged[:, blend_start + cut:blend_end] = right_arr[:, blend_start + cut:blend_end]

        elif blend_mode == "linear":
            # Linear gradient
            alpha = np.linspace(1.0, 0.0, bw)[np.newaxis, :, np.newaxis]
            merged[:, blend_start:blend_end] = (
                left_arr[:, blend_start:blend_end] * alpha +
                right_arr[:, blend_start:blend_end] * (1 - alpha)
            )

        else:  # smooth (sigmoid)
            # Smooth sigmoid for natural seam
            x = np.linspace(-4, 4, bw)
            alpha = 1.0 / (1.0 + np.exp(x))
            alpha = alpha[np.newaxis, :, np.newaxis]
            merged[:, blend_start:blend_end] = (
                left_arr[:, blend_start:blend_end] * alpha +
                right_arr[:, blend_start:blend_end] * (1 - alpha)
            )

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
