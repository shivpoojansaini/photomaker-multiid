# generation.py

import random
import numpy as np
import torch
from PIL import Image
from typing import Tuple, List, Optional
from diffusers.utils import load_image
from .style_template import styles
from .face_utils import extract_left_right_faces
from .prompt_splitter import split_prompt
from .image_merger import merge_images, side_by_side

MAX_SEED = np.iinfo(np.int32).max


def validate_trigger_word(pipe, prompt):
    token_id = pipe.tokenizer.convert_tokens_to_ids(pipe.trigger_word)
    ids = pipe.tokenizer.encode(prompt)

    if token_id not in ids:
        raise ValueError(f"Trigger word '{pipe.trigger_word}' missing in prompt: {prompt}")

    if ids.count(token_id) > 1:
        raise ValueError(f"Multiple trigger words '{pipe.trigger_word}' found in prompt: {prompt}")


def apply_style(style_name, positive, negative):
    default = "Photographic (Default)"
    p, n = styles.get(style_name, styles[default])
    return p.replace("{prompt}", positive), n + " " + negative


def generate_multi_identity(
    pipe,
    face_detector,
    input_image_path: str,
    prompt: str,
    seed: Optional[int] = None,
    style_name: str = "Photographic (Default)",
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    num_steps: int = 50,
    style_strength_ratio: float = 20,
    guidance_scale: float = 5.0,
    overlap_ratio: float = 0.3,
    merge_mode: str = "blend",  # "blend", "side_by_side", or "separate"
) -> Tuple[Image.Image, Optional[Image.Image], Optional[Image.Image], int]:
    """
    Generate multi-identity image from a SINGLE prompt.

    This function:
    1. Parses the single prompt into left/right prompts
    2. Extracts face embeddings from input image
    3. Generates two separate images (one per identity)
    4. Merges them with overlap blending

    Args:
        pipe: PhotoMaker pipeline
        face_detector: InsightFace detector
        input_image_path: Path to input image with 2 people
        prompt: Single combined prompt like "left person as teacher, right person as doctor"
        seed: Random seed (optional)
        style_name: Style template name
        negative_prompt: Negative prompt
        width: Output width
        height: Output height
        num_steps: Number of inference steps
        style_strength_ratio: Style strength (controls merge step)
        guidance_scale: CFG scale
        overlap_ratio: Overlap ratio for blending (0.1-0.5)
        merge_mode: "blend" (merged), "side_by_side", or "separate" (return both)

    Returns:
        Tuple of (merged_image, left_image, right_image, seed)
        - If merge_mode="blend": merged_image is the blended result
        - If merge_mode="side_by_side": merged_image is concatenated
        - If merge_mode="separate": merged_image is None, returns individual images

    Example:
        merged, left, right, seed = generate_multi_identity(
            pipe, detector,
            input_image_path="couple.jpg",
            prompt="left person as astronaut, right person as doctor"
        )
        merged.save("output.png")
    """
    # Load input image
    input_image = load_image(input_image_path)

    # Parse single prompt into left/right
    left_prompt, right_prompt = split_prompt(prompt, pipe.trigger_word)

    print(f"  Parsed prompts:")
    print(f"    Left:  {left_prompt}")
    print(f"    Right: {right_prompt}")

    # Extract identity faces (cropped images + embeddings)
    (left_face_img, id_left), (right_face_img, id_right) = extract_left_right_faces(
        face_detector, input_image
    )
    print(f"  Extracted face crops: left={left_face_img.size}, right={right_face_img.size}")

    # Seed
    seed = seed if seed is not None else random.randint(0, MAX_SEED)

    # Merge step
    start_merge_step = int(float(style_strength_ratio) / 100 * num_steps)
    start_merge_step = min(start_merge_step, 30)

    # Apply style
    prompt_left_styled, neg_left = apply_style(style_name, left_prompt, negative_prompt)
    prompt_right_styled, neg_right = apply_style(style_name, right_prompt, negative_prompt)

    # Validate trigger words
    validate_trigger_word(pipe, prompt_left_styled)
    validate_trigger_word(pipe, prompt_right_styled)

    # Generate LEFT identity image (using LEFT face crop only)
    print(f"  Generating left identity...")
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    left_result = pipe(
        prompt=prompt_left_styled,
        width=width,
        height=height,
        input_id_images=[left_face_img],  # Use cropped left face only
        negative_prompt=neg_left,
        num_images_per_prompt=1,
        num_inference_steps=num_steps,
        start_merge_step=start_merge_step,
        generator=generator,
        guidance_scale=guidance_scale,
        id_embeds=id_left,
    ).images[0]

    # Generate RIGHT identity image (using RIGHT face crop only)
    print(f"  Generating right identity...")
    generator = torch.Generator(device=pipe.device).manual_seed(seed)  # Reset for consistency
    right_result = pipe(
        prompt=prompt_right_styled,
        width=width,
        height=height,
        input_id_images=[right_face_img],  # Use cropped right face only
        negative_prompt=neg_right,
        num_images_per_prompt=1,
        num_inference_steps=num_steps,
        start_merge_step=start_merge_step,
        generator=generator,
        guidance_scale=guidance_scale,
        id_embeds=id_right,
    ).images[0]

    # Merge results
    if merge_mode == "blend":
        # Convert ratio to percent if needed (overlap_ratio was 0.0-1.0, now we use 5-20%)
        overlap_pct = overlap_ratio * 100 if overlap_ratio <= 1.0 else overlap_ratio
        # Clamp to 5-20% range, or use random if outside range
        if overlap_pct < 5 or overlap_pct > 20:
            overlap_pct = None  # Will use random 5-20%
        print(f"  Merging with ~{overlap_pct or '5-20'}% overlap zone...")
        merged = merge_images(left_result, right_result, overlap_percent=overlap_pct)
        return merged, left_result, right_result, seed

    elif merge_mode == "side_by_side":
        print(f"  Creating side-by-side view...")
        merged = side_by_side(left_result, right_result, gap=10)
        return merged, left_result, right_result, seed

    else:  # separate
        return None, left_result, right_result, seed


def generate_images(
    pipe,
    face_detector,
    input_image_path,
    left_prompt,
    right_prompt,
    seed,
    style_name,
    negative_prompt,
    width,
    height,
    num_outputs,
    num_steps,
    style_strength_ratio,
    guidance_scale
):
    """
    Original function - kept for backward compatibility.
    Generates separate left and right images without merging.
    """
    # Load input image
    input_image = load_image(input_image_path)

    # Extract identity faces (cropped images + embeddings)
    (left_face_img, id_left), (right_face_img, id_right) = extract_left_right_faces(
        face_detector, input_image
    )

    # Seed
    seed = seed if seed is not None else random.randint(0, MAX_SEED)
    generator = torch.Generator(device=pipe.device).manual_seed(seed)

    # Merge step
    start_merge_step = int(float(style_strength_ratio) / 100 * num_steps)
    start_merge_step = min(start_merge_step, 30)

    left_results = []
    right_results = []

    # LEFT FACE
    validate_trigger_word(pipe, left_prompt)
    prompt_left, neg_left = apply_style(style_name, left_prompt, negative_prompt)

    imgs_left = pipe(
        prompt=prompt_left,
        width=width,
        height=height,
        input_id_images=[left_face_img],  # Use cropped left face
        negative_prompt=neg_left,
        num_images_per_prompt=num_outputs,
        num_inference_steps=num_steps,
        start_merge_step=start_merge_step,
        generator=generator,
        guidance_scale=guidance_scale,
        id_embeds=id_left,
    ).images

    left_results.append((left_prompt, imgs_left))

    # RIGHT FACE
    validate_trigger_word(pipe, right_prompt)
    prompt_right, neg_right = apply_style(style_name, right_prompt, negative_prompt)

    imgs_right = pipe(
        prompt=prompt_right,
        width=width,
        height=height,
        input_id_images=[right_face_img],  # Use cropped right face
        negative_prompt=neg_right,
        num_images_per_prompt=num_outputs,
        num_inference_steps=num_steps,
        start_merge_step=start_merge_step,
        generator=generator,
        guidance_scale=guidance_scale,
        id_embeds=id_right,
    ).images

    right_results.append((right_prompt, imgs_right))

    return left_results, right_results, seed
