import os
import re
import requests
from io import BytesIO

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import Config
from train import CloudDeepLabV3Plus

# ────────────────────── 0) FUNCIONES REMOTAS ──────────────────────
def open_remote_image(url: str) -> Image.Image:
    r = requests.get(url); r.raise_for_status()
    return Image.open(BytesIO(r.content))

def remote_exists(url: str) -> bool:
    return requests.head(url).status_code == 200

# ──────────────────── 1) PALETA & AUXILIARES ────────────────────
PALETTE = [
    (255,255,255), (128,0,0), (0,128,0),
    (255,255,0),   (0,0,0),   (128,0,128)
]
FLAT_PAL = [c for rgb in PALETTE for c in rgb]
CLASS_NAMES = ["Fondo","Lengua de vaca","Diente de león","Kikuyo","Otro","Papa"]

def rgb_to_idx(rgb_arr: np.ndarray, palette: list[tuple[int,int,int]]) -> np.ndarray:
    idx = np.zeros(rgb_arr.shape[:2], dtype=np.uint8)
    for i, color in enumerate(palette):
        idx[np.all(rgb_arr == color, axis=-1)] = i
    return idx

# ──────────────────── 2) CARGA DE GT ────────────────────
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
            cand = f"{os.path.dirname(image_url).rsplit('/',1)[0]}/{d}{base}{ext}"
            if remote_exists(cand):
                return cand
    return None

def load_gt(mask_url: str | None, size: tuple[int,int]) -> Image.Image:
    if mask_url is None:
        return Image.new("RGB", size, (0,0,0))
    m = open_remote_image(mask_url)
    if m.mode in ("P","L","I"):
        idx = np.array(m, dtype=np.uint8)
    else:
        idx = rgb_to_idx(np.array(m.convert("RGB")), PALETTE)
    gt = Image.fromarray(idx, mode="P"); gt.putpalette(FLAT_PAL)
    return gt.convert("RGB").resize(size, Image.NEAREST)

# ──────────────────── 3) LISTA DE IMÁGENES ────────────────────
image_urls = [
    f"{BASE_URL}/images/5-113m3_jpg.rf.1a908ea089918e172ac9b1cfbc81b590.jpg",
    f"{BASE_URL}/images/101_jpg.rf.2a2a92bdf083fea463b938aa1f3e6bbf.jpg",
    f"{BASE_URL}/images/118_jpg.rf.eceeb04c2e33998be1c3ded4e4bd0fdd.jpg",
    f"{BASE_URL}/images/137_jpg.rf.6980a8e200cb1d6a3471c93debb03d04.jpg",
]

# ──────────────────── 4) MODELO + TRANSFORMS ────────────────────
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

# ───────────────────── 5) INFERENCIAS ─────────────────────
results = []
for url in image_urls:
    orig = open_remote_image(url).convert("RGB")

    gt_small = load_gt(find_mask(url), size=(Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT))
    gt_idx_small = rgb_to_idx(np.array(gt_small), PALETTE)

    tensor = tfm(image=np.array(orig))["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        logits = logits[0] if isinstance(logits, tuple) else logits
        pred_idx = torch.argmax(logits,1).squeeze().cpu().numpy()

    gt_mask = load_gt(find_mask(url), orig.size)
    pred_img = Image.fromarray(pred_idx.astype(np.uint8), mode="P")
    pred_img.putpalette(FLAT_PAL)
    pred_rgb = pred_img.convert("RGB").resize(orig.size, Image.NEAREST)

    results.append((orig, gt_mask, pred_rgb, gt_idx_small, pred_idx))

# ───────────────────── 6) PREPARACIÓN DE LEYENDA ─────────────────────
handles = [
    Patch(facecolor=np.array(rgb)/255., edgecolor="black", label=name)
    for rgb, name in zip(PALETTE, CLASS_NAMES)
]

# ─────────────────── 7) VISUALIZACIÓN Y GUARDADO ────────────────────
output_dir = "/content/drive/MyDrive/colab/"
os.makedirs(output_dir, exist_ok=True)

for idx, (orig, gt_rgb, pred_rgb, gt_idx_small, pred_idx) in enumerate(results, start=1):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.subplots_adjust(top=0.85, bottom=0.25, wspace=0.3)

    # Mostrar cada imagen con bordes y títulos destacados
    for ax, img, title in zip(
        axes,
        (orig, gt_rgb, pred_rgb),
        ("Imagen Original", "Máscara Real (GT)", "Predicción del Modelo")
    ):
        interp = 'nearest' if title != "Imagen Original" else None
        ax.imshow(img, interpolation=interp)
        ax.set_title(title, fontsize=16, fontweight='bold', pad=12)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1)
            spine.set_edgecolor('black')

    # Calcular IoU por clase
    ious = []
    for cls in range(len(CLASS_NAMES)):
        m_gt = (gt_idx_small == cls)
        m_pred = (pred_idx == cls)
        inter = np.logical_and(m_gt, m_pred).sum()
        uni = np.logical_or(m_gt, m_pred).sum()
        ious.append(inter / uni if uni > 0 else 0.0)

    # Leyenda centrada abajo
    fig.legend(
        handles=handles,
        title='Leyenda de clases',
        title_fontsize=14,
        fontsize=12,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        frameon=False
    )

    # Tabla de IoU debajo de las imágenes
    table_data = [[CLASS_NAMES[i], f"{ious[i]:.2f}"] for i in range(len(CLASS_NAMES))]
    tbl = fig.table(
        cellText=table_data,
        colLabels=["Clase", "IoU"],
        cellLoc='center',
        loc='bottom',
        colColours=["#f1f1f2"] * 2
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1, 1.5)

    # Guardar y mostrar
    save_path = os.path.join(output_dir, f"fila_prediccion_{idx}.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
