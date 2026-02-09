# face_utils.py

import numpy as np
import torch
from PIL import Image
from photomaker import FaceAnalysis2, analyze_faces


def load_face_detector(device):
    print("Loading face detector...")
    providers = ["CUDAExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]

    detector = FaceAnalysis2(
        providers=providers,
        allowed_modules=["detection", "recognition"]
    )
    detector.prepare(ctx_id=0, det_size=(640, 640))
    return detector


def crop_face_image(pil_image, bbox, expansion=1.8):
    """
    Crop a face region from the image with expansion for head/shoulders.

    Args:
        pil_image: PIL Image
        bbox: [x1, y1, x2, y2] face bounding box
        expansion: Factor to expand the crop (1.8 = 80% larger than face)

    Returns:
        Cropped PIL Image
    """
    img_w, img_h = pil_image.size
    x1, y1, x2, y2 = bbox

    # Calculate face dimensions
    face_w = x2 - x1
    face_h = y2 - y1
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    # Expand the crop region
    new_w = face_w * expansion
    new_h = face_h * expansion

    # Calculate crop bounds (ensure within image)
    crop_x1 = max(0, int(center_x - new_w / 2))
    crop_y1 = max(0, int(center_y - new_h / 2))
    crop_x2 = min(img_w, int(center_x + new_w / 2))
    crop_y2 = min(img_h, int(center_y + new_h / 2))

    return pil_image.crop((crop_x1, crop_y1, crop_x2, crop_y2))


def extract_left_right_faces(face_detector, pil_image, expansion=1.8):
    """
    Extract left and right face images with their embeddings.

    Args:
        face_detector: InsightFace detector
        pil_image: PIL Image with 2 people
        expansion: Crop expansion factor

    Returns:
        Tuple of ((left_crop, left_embed), (right_crop, right_embed))
        - left_crop/right_crop: PIL Images of cropped faces
        - left_embed/right_embed: torch tensors of shape [1, 512]
    """
    img_array = np.array(pil_image)[:, :, ::-1]  # RGB -> BGR for InsightFace

    faces = analyze_faces(face_detector, img_array)
    if len(faces) < 2:
        raise ValueError(f"Need at least two faces in the input image. Found: {len(faces)}")

    # Sort by x-coordinate (left to right)
    faces_sorted = sorted(faces, key=lambda f: f["bbox"][0])
    left_face, right_face = faces_sorted[:2]

    # Crop face images
    left_crop = crop_face_image(pil_image, left_face["bbox"], expansion)
    right_crop = crop_face_image(pil_image, right_face["bbox"], expansion)

    # Get embeddings
    emb_left = torch.from_numpy(left_face["embedding"]).float()
    emb_right = torch.from_numpy(right_face["embedding"]).float()

    return (
        (left_crop, torch.stack([emb_left])),
        (right_crop, torch.stack([emb_right]))
    )


def extract_left_right_embeddings(face_detector, pil_image):
    """
    Legacy function - returns only embeddings.
    Use extract_left_right_faces() for full functionality.
    """
    (_, emb_left), (_, emb_right) = extract_left_right_faces(face_detector, pil_image)
    return emb_left, emb_right
