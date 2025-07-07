import os
import re
import requests
from io import BytesIO

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from pathlib import Path
from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import Config
from train_test4 import CloudDeepLabV3Plus  # Cámbiala si vive en otro módulo

# ─────────────────────────── 0) FUNCIONES PARA IMÁGENES REMOTAS ───────────────────
def open_remote_image(url: str) -> Image.Image:
    r = requests.get(url)
    r.raise_for_status()
    return Image.open(BytesIO(r.content))

def remote_exists(url: str) -> bool:
    r = requests.head(url)
    return r.status_code == 200

# ─────────────────────────── 1) PALETA Y FUNCIONES AUXILIARES ─────────────────────
PALETTE = [
    (255,255,255), (128,0,0), (0,128,0),
    (255,255,0),   (0,0,0),   (128,0,128)
]
FLAT_PAL = [c for rgb in PALETTE for c in rgb]

def rgb_to_idx(rgb_arr: np.ndarray, palette: list[tuple[int,int,int]]) -> np.ndarray:
    idx = np.zeros(rgb_arr.shape[:2], dtype=np.uint8)
    for i, color in enumerate(palette):
        idx[np.all(rgb_arr == color, axis=-1)] = i
    return idx

# ─────────────────────────── 2) BÚSQUEDA DE MÁSCARA GT REMOTA ─────────────────────
BASE_URL = "https://raw.githubusercontent.com/JorgePazos-git/Dataset-of-weeds-in-potato-crops-in-the-province-of-Carchi-and-Imbabura-in-/refs/heads/main/Balanced/val"

def find_mask(image_url: str) -> str | None:
    base = re.sub(r"\.(jpg|jpeg)$", "", os.path.basename(image_url), flags=re.I)
    dirs = ["", "labels/", "masks/"]
    exts = ["_mask.png", ".png"]
    for d in dirs:
        for ext in exts:
            candidate = f"{os.path.dirname(image_url).rsplit('/',1)[0]}/{d}{base}{ext}"
            if remote_exists(candidate):
                return candidate
    return None

def load_gt(mask_url: str | None, size: tuple[int,int]) -> Image.Image:
    if mask_url is None:
        return Image.new("RGB", size, (0,0,0))
    m = open_remote_image(mask_url)
    if m.mode in ("P", "L", "I"):
        idx = np.array(m, dtype=np.uint8)
    else:
        idx = rgb_to_idx(np.array(m.convert("RGB")), PALETTE)
    gt = Image.fromarray(idx, mode="P")
    gt.putpalette(FLAT_PAL)
    return gt.convert("RGB").resize(size, Image.NEAREST)

# ─────────────────────────── 3) LISTA DE IMÁGENES ────────────────────────────────
image_urls = [
    f"{BASE_URL}/images/DJI_0058-JPG_1500_2000_JPG.rf.dfa17ea56a5a4a7aebd29c2b33de2522.jpg",
    f"{BASE_URL}/images/DJI_0055-JPG_2250_1750_JPG.rf.1da573f04d505c757d408807e17e1be8.jpg",
    f"{BASE_URL}/images/bright_5-221m3_jpg.rf.9a48144a508a0e410fa0225c37c21474.jpg",
    f"{BASE_URL}/images/67_jpg.rf.841de14d82ab29ae699603779f003823.jpg",
]

# ─────────────────────────── 4) MODELO Y TRANSFORMACIÓN ──────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = torch.load(
    "/content/drive/MyDrive/colab/cultivos_deeplab_final.pt",
    map_location=device, weights_only=False
)
model.eval()

tfm = A.Compose([
    A.Resize(Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH),
    A.Normalize(mean=(0,0,0), std=(1,1,1), max_pixel_value=255.0),
    ToTensorV2(),
])

# ─────────────────────────── 5) INFERENCIAS Y VISUALIZACIÓN ──────────────────────
results = []
for url in image_urls:
    original = open_remote_image(url).convert("RGB")
    gt_mask  = load_gt(find_mask(url), original.size)

    tensor = tfm(image=np.array(original))["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        logits = logits[0] if isinstance(logits, tuple) else logits
        pred   = torch.argmax(logits, 1).squeeze().cpu().numpy()

    pred_img = Image.fromarray(pred.astype(np.uint8), mode="P")
    pred_img.putpalette(FLAT_PAL)
    pred_rgb = pred_img.convert("RGB").resize(original.size, Image.NEAREST)

    results.append((original, gt_mask, pred_rgb))

n = len(results)
fig, axs = plt.subplots(n, 3, figsize=(15, 5*n), constrained_layout=True)
for i, (orig, gt, pred) in enumerate(results):
    row = axs[i] if n > 1 else axs
    for ax, img, title in zip(row, (orig, gt, pred),
                              ("Imagen original", "Máscara GT", "Predicción")):
        ax.imshow(img)
        ax.set_title(title, fontsize=12)
        ax.axis("off")

# ───── LISTA DE NOMBRES DE CLASE ───────────────────────────────────────────
CLASS_NAMES = [
    "Fondo", "Lengua de vaca", "Diente de león",
    "Kikuyo", "Otro", "Papa",
]
assert len(CLASS_NAMES) == len(PALETTE), "‽ Mismos elementos en PALETTE y CLASS_NAMES"

# ───── LEYENDA DE COLORES POR CLASE ────────────────────────────────────────
handles = [
    Patch(facecolor=np.array(rgb)/255.0,
          edgecolor="black",
          label=CLASS_NAMES[i])
    for i, rgb in enumerate(PALETTE)
]
fig.legend(
    handles=handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=len(PALETTE),
    frameon=False,
    fontsize=11
)

plt.savefig("all_predictions_grid.png", dpi=300, bbox_inches="tight")
plt.show()

if __name__ == "__main__":
    pass
