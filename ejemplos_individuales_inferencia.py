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
from train import CloudDeepLabV3Plus  # Cámbiala si está en otro módulo

# ─────────────────────────── 0) FUNCIONES PARA IMÁGENES REMOTAS ───────────────────
def open_remote_image(url: str) -> Image.Image:
    r = requests.get(url)
    r.raise_for_status()
    return Image.open(BytesIO(r.content))

def remote_exists(url: str) -> bool:
    return requests.head(url).status_code == 200

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
BASE_URL = (
    "https://raw.githubusercontent.com/"
    "JorgePazos-git/Dataset-of-weeds-in-potato-crops-in-the-province-of-Carchi-and-Imbabura-in-"
    "/refs/heads/main/Balanced/train"
)

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
    f"{BASE_URL}/images/5-113m3_jpg.rf.1a908ea089918e172ac9b1cfbc81b590.jpg",
    f"{BASE_URL}/images/101_jpg.rf.2a2a92bdf083fea463b938aa1f3e6bbf.jpg",
    f"{BASE_URL}/images/118_jpg.rf.eceeb04c2e33998be1c3ded4e4bd0fdd.jpg",
    f"{BASE_URL}/images/137_jpg.rf.6980a8e200cb1d6a3471c93debb03d04.jpg",
]

# ─────────────────────────── 4) MODELO Y TRANSFORMACIÓN ──────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(
    "/content/drive/MyDrive/colab/cultivos_deeplab_final.pt",
    map_location=device, weights_only=False
)
model.eval()

tfm = A.Compose([
    A.Resize(Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH),
    A.Normalize(mean=(0,0,0), std=(1,1,1), max_pixel_value=255.0),
    ToTensorV2(),
])

# ─────────────────────────── 5) INFERENCIAS ───────────────────────────────────────
results = []
for url in image_urls:
    orig = open_remote_image(url).convert("RGB")
    gt_mask = load_gt(find_mask(url), orig.size)

    tensor = tfm(image=np.array(orig))["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        logits = logits[0] if isinstance(logits, tuple) else logits
        pred_idx = torch.argmax(logits, 1).squeeze().cpu().numpy()

    # convertir a RGB para visualizar
    pred_img = Image.fromarray(pred_idx.astype(np.uint8), mode="P")
    pred_img.putpalette(FLAT_PAL)
    pred_rgb = pred_img.convert("RGB").resize(orig.size, Image.NEAREST)

    results.append((orig, gt_mask, pred_rgb, pred_idx))

# ─────────────────────────── 6) LEYENDA DE COLORES POR CLASE ────────────────────
CLASS_NAMES = [
    "Fondo", "Lengua de vaca", "Diente de león",
    "Kikuyo", "Otro", "Papa",
]
assert len(CLASS_NAMES) == len(PALETTE), "PALETTE y CLASS_NAMES deben coincidir en longitud"

handles = [
    Patch(facecolor=np.array(rgb)/255.0, edgecolor="black", label=CLASS_NAMES[i])
    for i, rgb in enumerate(PALETTE)
]

# ─────────────────── 7) VISUALIZACIÓN, IoU POR CLASE Y ANOTACIONES ─────────────────
output_dir = "/content/drive/MyDrive/colab/"
os.makedirs(output_dir, exist_ok=True)

for idx, (orig, gt, pred, pred_idx) in enumerate(results, start=1):
    # reconstruir índices de GT
    gt_idx = rgb_to_idx(np.array(gt), PALETTE)

    # calcular IoU por clase
    ious = []
    for c in range(len(PALETTE)):
        inter = np.logical_and(gt_idx == c, pred_idx == c).sum()
        uni   = np.logical_or (gt_idx == c, pred_idx == c).sum()
        ious.append(inter/uni if uni > 0 else 0.0)

    # crear figura
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=False)
    fig.subplots_adjust(top=0.75)

    # mostrar cada imagen
    for ax, img, title in zip(
        axes,
        (orig, gt, pred),
        ("Imagen original", "Máscara GT", "Predicción")
    ):
        ax.imshow(img)
        ax.set_title(title, fontsize=12, pad=10)
        ax.axis("off")

    # anotaciones IoU en la predicción
    ax_pred = axes[2]
    for c, iou in enumerate(ious):
        if iou <= 0:
            continue
        ys, xs = np.where(pred_idx == c)
        if ys.size == 0:
            continue
        y0, x0 = ys.mean(), xs.mean()
        ax_pred.annotate(
            f"{CLASS_NAMES[c]} = {iou:.3f}",
            xy=(x0, y0), xytext=(x0 + 30, y0 + 30),
            color="white", fontsize=10,
            arrowprops=dict(arrowstyle="->", color="white", lw=1)
        )

    # leyenda de colores
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=len(PALETTE),
        frameon=False,
        fontsize=11
    )

    # guardar y mostrar
    save_path = os.path.join(output_dir, f"fila_prediccion_{idx}.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

if __name__ == "__main__":
    pass
