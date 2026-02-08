#!/usr/bin/env python3
"""
Multi-Identity PhotoMaker with Single Prompt Interface

Usage:
    python run_multi_identity.py --input_image couple.jpg --prompt "left person as teacher, right person as doctor"

This script:
1. Takes a single natural language prompt
2. Automatically splits it into left/right prompts
3. Generates separate images for each identity
4. Merges them with smooth overlap blending
"""

import argparse
from pathlib import Path
import sys

# Add the Code directory to path
sys.path.insert(0, str(Path(__file__).parent / "Code"))

from PhotoMaker_Extensions.pipeline_loader import load_pipeline, get_device
from PhotoMaker_Extensions.face_utils import load_face_detector
from PhotoMaker_Extensions.generation import generate_multi_identity
from PhotoMaker_Extensions.watermark import add_watermark
from PhotoMaker_Extensions.invisible_watermark.utils import encode_watermark
from PhotoMaker_Extensions.config import (
    OUTPUT_DIR,
    STYLE_NAME,
    NEGATIVE_PROMPT,
    OUTPUT_WIDTH,
    OUTPUT_HEIGHT,
    NUM_STEPS,
    STYLE_STRENGTH_RATIO,
    GUIDANCE_SCALE,
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate multi-identity images with a single prompt"
    )
    parser.add_argument(
        "--input_image", type=str, required=True,
        help="Path to input image with 2 people"
    )
    parser.add_argument(
        "--prompt", type=str, required=True,
        help="Single prompt like 'left person as teacher, right person as doctor'"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output file path (default: auto-generated in OUTPUT_DIR)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--overlap", type=float, default=0.3,
        help="Overlap ratio for blending (0.1-0.5, default: 0.3)"
    )
    parser.add_argument(
        "--merge_mode", type=str, default="blend",
        choices=["blend", "side_by_side", "separate"],
        help="How to combine outputs: blend, side_by_side, or separate"
    )
    parser.add_argument(
        "--no_watermark", action="store_true",
        help="Skip watermark"
    )
    parser.add_argument(
        "--width", type=int, default=OUTPUT_WIDTH,
        help=f"Output width (default: {OUTPUT_WIDTH})"
    )
    parser.add_argument(
        "--height", type=int, default=OUTPUT_HEIGHT,
        help=f"Output height (default: {OUTPUT_HEIGHT})"
    )
    parser.add_argument(
        "--steps", type=int, default=NUM_STEPS,
        help=f"Inference steps (default: {NUM_STEPS})"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Multi-Identity PhotoMaker")
    print("=" * 60)

    # Setup output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    device = get_device()
    print(f"Using device: {device}")

    pipe = load_pipeline(device)
    face_detector = load_face_detector(device)

    # Generate
    print(f"\nInput: {args.input_image}")
    print(f"Prompt: {args.prompt}")
    print(f"Merge mode: {args.merge_mode}")

    merged_img, left_img, right_img, seed = generate_multi_identity(
        pipe=pipe,
        face_detector=face_detector,
        input_image_path=args.input_image,
        prompt=args.prompt,
        seed=args.seed,
        style_name=STYLE_NAME,
        negative_prompt=NEGATIVE_PROMPT,
        width=args.width,
        height=args.height,
        num_steps=args.steps,
        style_strength_ratio=STYLE_STRENGTH_RATIO,
        guidance_scale=GUIDANCE_SCALE,
        overlap_ratio=args.overlap,
        merge_mode=args.merge_mode,
    )

    print(f"\nSeed: {seed}")

    # Watermark
    bitstring = [1, 0, 1, 1, 0, 1, 0, 1] * 8

    def apply_watermarks(img):
        if args.no_watermark:
            return img
        img = encode_watermark(img, bitstring)
        img = add_watermark(img)
        return img

    # Save outputs
    if args.merge_mode == "separate":
        # Save individual images
        left_path = output_dir / f"left_seed{seed}.png"
        right_path = output_dir / f"right_seed{seed}.png"

        apply_watermarks(left_img).save(left_path)
        apply_watermarks(right_img).save(right_path)

        print(f"Saved: {left_path}")
        print(f"Saved: {right_path}")

    else:
        # Save merged image
        if args.output:
            output_path = Path(args.output)
        else:
            mode_suffix = "merged" if args.merge_mode == "blend" else "sidebyside"
            output_path = output_dir / f"{mode_suffix}_seed{seed}.png"

        apply_watermarks(merged_img).save(output_path)
        print(f"Saved: {output_path}")

        # Also save individual images for reference
        left_path = output_dir / f"left_seed{seed}.png"
        right_path = output_dir / f"right_seed{seed}.png"
        left_img.save(left_path)
        right_img.save(right_path)
        print(f"Saved individual: {left_path}, {right_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
