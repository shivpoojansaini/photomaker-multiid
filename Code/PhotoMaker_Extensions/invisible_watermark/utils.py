# PhotoMaker_Extensions/invisible_watermark/utils.py

import torch
from pathlib import Path
from PIL import Image
import torchvision.transforms as T

# Global flag to track if watermark is available
WATERMARK_AVAILABLE = False

try:
    from .encoder import WatermarkEncoder
    from .decoder import WatermarkDecoder

    BIT_LENGTH = 64

    encoder = WatermarkEncoder(bit_length=BIT_LENGTH)
    decoder = WatermarkDecoder(bit_length=BIT_LENGTH)

    # Try loading trained weights
    encoder_path = Path(__file__).parent / "encoder_best.pth"
    decoder_path = Path(__file__).parent / "decoder_best.pth"

    if encoder_path.exists() and decoder_path.exists():
        try:
            encoder.load_state_dict(torch.load(encoder_path, map_location="cpu"))
            decoder.load_state_dict(torch.load(decoder_path, map_location="cpu"), strict=False)
            encoder.eval()
            decoder.eval()
            WATERMARK_AVAILABLE = True
            print("Loaded trained invisible watermark models.")
        except RuntimeError as e:
            print(f"Warning: Could not load watermark models (architecture mismatch): {e}")
            print("Invisible watermark will be skipped.")
    else:
        print("Warning: Watermark model files not found. Invisible watermark will be skipped.")

except ImportError as e:
    print(f"Warning: Could not import watermark modules: {e}")
    encoder = None
    decoder = None

to_tensor = T.ToTensor()
to_pil = T.ToPILImage()


def encode_watermark(pil_img, bitstring):
    """Encode watermark into image. Returns original if watermark unavailable."""
    if not WATERMARK_AVAILABLE or encoder is None:
        return pil_img

    try:
        img = to_tensor(pil_img).unsqueeze(0)
        bits = torch.tensor(bitstring).float().unsqueeze(0)
        wm = encoder(img, bits)
        return to_pil(wm.squeeze(0))
    except Exception as e:
        print(f"Warning: Watermark encoding failed: {e}")
        return pil_img


def decode_watermark(pil_img):
    """Decode watermark from image. Returns None if watermark unavailable."""
    if not WATERMARK_AVAILABLE or decoder is None:
        return None

    try:
        img = to_tensor(pil_img).unsqueeze(0)
        bits = decoder(img).detach().squeeze(0)
        return (bits > 0.5).int().tolist()
    except Exception as e:
        print(f"Warning: Watermark decoding failed: {e}")
        return None

