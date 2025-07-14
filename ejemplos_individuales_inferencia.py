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
    base = re.sub(r"\.(jpg|jpeg)$","",os.path.basename(image_url),flags=re.I)
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

# ───────────────────── 6) LEYENDA ─────────────────────
CLASS_NAMES = ["Fondo","Lengua de vaca","Diente de león","Kikuyo","Otro","Papa"]
handles = [Patch(facecolor=np.array(rgb)/255., edgecolor="black", label=CLASS_NAMES[i])
           for i,rgb in enumerate(PALETTE)]

# ───────────────────── 7) PLOT + IoU ─────────────────────
output_dir = "/content/drive/MyDrive/colab/"
os.makedirs(output_dir, exist_ok=True)

def rects_intersect(a, b):
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])

for idx, (orig, gt_mask, pred_rgb, gt_idx, pred_idx) in enumerate(results,1):
    ious = []
    for c in range(len(PALETTE)):
        inter = np.logical_and(gt_idx==c, pred_idx==c).sum()
        uni   = np.logical_or(gt_idx==c, pred_idx==c).sum()
        ious.append(inter/uni if uni>0 else 0.0)

    print(f"\nIoU imagen {idx}:")
    for c,iou in enumerate(ious):
        print(f"  {CLASS_NAMES[c]:<15}: {iou:.3f}")

    fig, axes = plt.subplots(1,3,figsize=(15,5))
    fig.subplots_adjust(top=0.75)
    for ax,img,title in zip(axes,(orig,gt_mask,pred_rgb),
                             ("Imagen original","Máscara GT","Predicción")):
        ax.imshow(img); ax.set_title(title,fontsize=12,pad=10); ax.axis("off")

    ax_pred = axes[2]
    h_small,w_small = pred_idx.shape
    h_big,w_big,_  = np.array(pred_rgb).shape
    sx,sy = w_big/w_small, h_big/h_small

    placed_boxes = []
    radius = 50

    # 1. Mucha más granularidad: 360 ángulos (paso ≈1°)
    base_angles = np.linspace(0, 2*np.pi, 360, endpoint=False)
    # 2. Los ordenas igual, poniendo primero los más horizontales
    angles = sorted(base_angles, key=lambda t: abs(np.sin(t)))

    fontsize = 10
    char_w = 7
    text_h = fontsize

    for c, iou in enumerate(ious):
        if iou <= 0: continue
        ys, xs = np.where(pred_idx==c)
        if ys.size == 0: continue

        y0_small, x0_small = np.median(ys), np.median(xs)
        x0, y0 = x0_small*sx, y0_small*sy

        text = f"{CLASS_NAMES[c]} = {iou:.3f}"
        text_w = len(text)*char_w

        for theta in angles:
            dx = np.cos(theta)*radius
            dy = np.sin(theta)*radius
            tx = x0 + dx
            ty = y0 + dy
            bbox = (tx, ty-text_h, tx+text_w, ty)
            if not any(rects_intersect(bbox, pb) for pb in placed_boxes):
                placed_boxes.append(bbox)
                ax_pred.annotate(
                    text,
                    xy=(x0, y0),
                    xytext=(tx, ty),
                    color="black", fontsize=fontsize, zorder=3,
                    arrowprops=dict(arrowstyle="->", color="black", lw=1, zorder=1),
                    clip_on=False
                )
                break

    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5,0.95),
               ncol=len(PALETTE), frameon=False, fontsize=11)

    save_path = os.path.join(output_dir, f"fila_prediccion_{idx}.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

if __name__=="__main__":
    pass
