# image_merger.py
# Merge two generated images (left identity, right identity) into a single output

import numpy as np
from PIL import Image
from typing import Tuple, Optional


def create_blend_mask(width: int, height: int, overlap_ratio: float = 0.3) -> np.ndarray:
    """
    Create a horizontal gradient mask for blending two images.

    The mask transitions from 1.0 (left) to 0.0 (right) in the overlap region.

    Args:
        width: Image width
        height: Image height
        overlap_ratio: Ratio of image width for the overlap/blend region (0.0-1.0)

    Returns:
        Gradient mask array of shape (height, width) with values 0.0-1.0
    """
    # Calculate overlap region
    overlap_width = int(width * overlap_ratio)
    center = width // 2

    # Start and end of gradient
    start = center - overlap_width // 2
    end = center + overlap_width // 2

    # Create horizontal gradient
    mask = np.zeros((height, width), dtype=np.float32)

    # Left region: fully left image (1.0)
    mask[:, :start] = 1.0

    # Overlap region: linear gradient
    if end > start:
        gradient = np.linspace(1.0, 0.0, end - start)
        mask[:, start:end] = gradient[np.newaxis, :]

    # Right region: fully right image (0.0)
    mask[:, end:] = 0.0

    return mask


def create_gaussian_blend_mask(
    width: int,
    height: int,
    overlap_ratio: float = 0.3,
    sigma_ratio: float = 0.15
) -> np.ndarray:
    """
    Create a smoother Gaussian-based blend mask.

    Args:
        width: Image width
        height: Image height
        overlap_ratio: Ratio of image width for the overlap region
        sigma_ratio: Gaussian sigma as ratio of width (controls smoothness)

    Returns:
        Smooth gradient mask array
    """
    center = width // 2
    sigma = width * sigma_ratio

    # Create x coordinates
    x = np.arange(width)

    # Gaussian-like smooth transition (using sigmoid for efficiency)
    # sigmoid centered at 'center' with smoothness controlled by sigma
    mask_1d = 1.0 / (1.0 + np.exp((x - center) / (sigma / 4)))

    # Expand to 2D
    mask = np.tile(mask_1d, (height, 1)).astype(np.float32)

    return mask


def merge_images(
    left_image: Image.Image,
    right_image: Image.Image,
    overlap_ratio: float = 0.3,
    blend_mode: str = "gaussian",
) -> Image.Image:
    """
    Merge two images with horizontal overlap blending.

    The left half of left_image and right half of right_image are combined
    with a smooth transition in the middle overlap region.

    Args:
        left_image: PIL Image generated with left identity
        right_image: PIL Image generated with right identity
        overlap_ratio: Width ratio for the blending overlap (0.1-0.5 recommended)
        blend_mode: "linear" or "gaussian" for blend transition type

    Returns:
        Merged PIL Image
    """
    # Ensure same size
    if left_image.size != right_image.size:
        right_image = right_image.resize(left_image.size, Image.LANCZOS)

    width, height = left_image.size

    # Convert to numpy arrays
    left_arr = np.array(left_image).astype(np.float32)
    right_arr = np.array(right_image).astype(np.float32)

    # Create blend mask
    if blend_mode == "gaussian":
        mask = create_gaussian_blend_mask(width, height, overlap_ratio)
    else:
        mask = create_blend_mask(width, height, overlap_ratio)

    # Expand mask for RGB channels
    mask_3d = mask[:, :, np.newaxis]

    # Blend: left * mask + right * (1 - mask)
    merged = left_arr * mask_3d + right_arr * (1 - mask_3d)

    # Convert back to PIL
    merged = np.clip(merged, 0, 255).astype(np.uint8)
    return Image.fromarray(merged)


def merge_with_face_regions(
    left_image: Image.Image,
    right_image: Image.Image,
    left_face_bbox: Tuple[float, float, float, float],
    right_face_bbox: Tuple[float, float, float, float],
    expansion_ratio: float = 1.5,
) -> Image.Image:
    """
    Merge images using face bounding boxes to determine the split point.

    The split point is calculated based on face positions to avoid
    cutting through faces.

    Args:
        left_image: PIL Image with left identity
        right_image: PIL Image with right identity
        left_face_bbox: (x1, y1, x2, y2) of left face
        right_face_bbox: (x1, y1, x2, y2) of right face
        expansion_ratio: Expand face region to include body

    Returns:
        Merged PIL Image
    """
    width, height = left_image.size

    # Calculate the gap between faces
    left_face_right_edge = left_face_bbox[2]  # x2 of left face
    right_face_left_edge = right_face_bbox[0]  # x1 of right face

    # Expand face regions to body
    left_region_end = min(left_face_right_edge * expansion_ratio, width * 0.6)
    right_region_start = max(right_face_left_edge / expansion_ratio, width * 0.4)

    # Calculate split point (middle of gap)
    split_point = (left_region_end + right_region_start) / 2
    split_ratio = split_point / width

    # Calculate overlap around split point
    overlap_width = abs(right_region_start - left_region_end) / width
    overlap_ratio = max(0.1, min(0.4, overlap_width))

    return merge_images(left_image, right_image, overlap_ratio, blend_mode="gaussian")


def side_by_side(
    left_image: Image.Image,
    right_image: Image.Image,
    gap: int = 0,
) -> Image.Image:
    """
    Simple side-by-side concatenation (no blending).

    Args:
        left_image: Left PIL Image
        right_image: Right PIL Image
        gap: Gap in pixels between images

    Returns:
        Combined PIL Image
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
