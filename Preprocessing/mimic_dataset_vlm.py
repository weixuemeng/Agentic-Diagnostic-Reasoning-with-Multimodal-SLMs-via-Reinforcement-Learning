import pandas as pd, ast, os
from pathlib import Path
from typing import Dict, Any
from PIL import Image
import numpy as np


def load_dicom(path: str) -> Image.Image:
    """Load DICOM → normalize → RGB PIL"""
    if pydicom is None:
        raise ValueError("pydicom not installed. Install: pip install pydicom")

    ds = pydicom.dcmread(path)
    arr = ds.pixel_array.astype(np.float32)

    # normalize to 0–255
    arr -= arr.min()
    arr /= (arr.max() + 1e-6)
    arr = (arr * 255).astype(np.uint8)

    return Image.fromarray(arr).convert("RGB")


def load_image(path: str) -> Image.Image:
    """Load JPG/PNG or DICOM"""
    p = path.lower()
    if p.endswith((".jpg")):
        return Image.open(path).convert("RGB")
    if p.endswith((".dcm",)): # we use this 
        return load_dicom(path)
    raise ValueError(f"Unsupported format: {path}")

def combine_pa_lat(images, target_h=768) -> Image.Image:
    """Concatenate images side-by-side"""
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
    Returns:
    {
        'image': PIL image,
        'reference': impression text,
        'prompt': text prompt,
        'study_id': id
    }
    """

    def __init__(self, csv_file: str, root: str):
        '''
        csv_file: path to the CSV file with 'image_paths' and 'impression' columns
        root: root directory for image paths (DICOM)
        '''
        self.df = pd.read_csv(csv_file)
        self.root = Path(root)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) -> Dict[str, Any]:
        row = self.df.iloc[idx]

        img_paths = ast.literal_eval(row["image_paths"])
        imgs = [] # Images list

        for p in img_paths[:2]:  # take first two
            full = self.root / p # Note: ../Data/Images/.. (need to set after download)
            try:
                imgs.append(load_image(str(full)))
            except Exception as e:
                print(f"[WARN] Image load failed {full}: {e}")

        if len(imgs) == 0:
            raise RuntimeError(f"No valid images for study {row['study_id']}")

        image = combine_pa_lat(imgs)

        return {
            "image": image,
            "reference": row["impression"],
            "prompt": "Two-view chest X-ray. Provide ONLY the Impression.",
            "study_id": row["study_id"]
        }
