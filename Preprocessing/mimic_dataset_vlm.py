import pandas as pd, ast, os, warnings
from pathlib import Path
from typing import Dict, Any
from PIL import Image
import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError


def load_dicom(path: str) -> Image.Image:
    """
    Load a DICOM file, normalize pixel data to 0–255, and return as RGB PIL Image.
    Safely handles incomplete or slightly corrupted files.
    """
    if pydicom is None:
        raise ValueError("pydicom not installed. Install it with: pip install pydicom")

    with warnings.catch_warnings():
        # Suppress expected-byte mismatch warnings
        warnings.filterwarnings(
            "ignore",
            message="The number of bytes of pixel data is less than expected"
        )

        try:
            ds = pydicom.dcmread(path, force=True)
        except InvalidDicomError:
            raise RuntimeError(f"Invalid DICOM file: {path}")
        except Exception as e:
            raise RuntimeError(f"Failed to read DICOM: {path} ({e})")

    # Extract and normalize pixel data
    try:
        arr = ds.pixel_array.astype(np.float32)
    except Exception as e:
        raise RuntimeError(f"Failed to decode pixel data for {path}: {e}")

    # Normalize to [0, 255]
    arr -= arr.min()
    arr /= (arr.max() + 1e-6)
    arr = (arr * 255).astype(np.uint8)

    # Convert to RGB PIL image
    return Image.fromarray(arr).convert("RGB")


def load_image(path: str) -> Image.Image:
    """
    Load JPG/PNG or DICOM image and return as RGB PIL Image.
    """
    p = path.lower()
    try:
        if p.endswith((".jpg", ".jpeg", ".png")):
            return Image.open(path).convert("RGB")
        elif p.endswith(".dcm"):
            return load_dicom(path)
        else:
            raise ValueError(f"Unsupported image format: {path}")
    except Exception as e:
        raise RuntimeError(f"Image load failed for {path}: {e}")


def combine_pa_lat(images, target_h=768) -> Image.Image:
    """
    Concatenate one or two X-ray views (PA + Lateral) side by side.
    Resizes each to the same height.
    """
    ims = []
    for img in images:
        w, h = img.size
        new_w = int(w * (target_h / h))
        ims.append(img.resize((new_w, target_h)))

    if len(ims) == 1:
        return ims[0]

    W = sum(i.size[0] for i in ims)
    canvas = Image.new("RGB", (W, target_h))
    x = 0
    for im in ims:
        canvas.paste(im, (x, 0))
        x += im.size[0]
    return canvas


class MIMICImpressionDataset:
    """
    Dataset for MIMIC-CXR-style multimodal impression generation.
    Returns a dict:
    {
        'image': PIL image,
        'reference': impression text,
        'prompt': instruction text,
        'study_id': unique identifier
    }
    """

    def __init__(self, csv_file: str, root: str):
        """
        Args:
            csv_file: Path to CSV containing 'image_paths' (list-like string) and 'impression' columns.
            root: Root directory where DICOMs or images are stored.
        """
        self.df = pd.read_csv(csv_file)
        self.root = Path(root)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) -> Dict[str, Any]:
        row = self.df.iloc[idx]

        try:
            img_paths = ast.literal_eval(row["image_paths"])
        except Exception as e:
            raise RuntimeError(f"Invalid image_paths format at index {idx}: {e}")

        imgs = []
        for p in img_paths[:2]:  # take first two if available
            full_path = self.root / p
            if not full_path.exists():
                print(f"[WARN] Missing file: {full_path}")
                continue
            try:
                imgs.append(load_image(str(full_path)))
            except Exception as e:
                print(f"[WARN] Image load failed {full_path}: {e}")

        if len(imgs) == 0:
            raise RuntimeError(f"No valid images found for study {row.get('study_id', idx)}")

        image = combine_pa_lat(imgs)

        return {
            "image": image,
            "reference": row["impression"],
            "prompt": "Two-view chest X-ray. Provide ONLY the Impression.",
            "study_id": row["study_id"]
        }
