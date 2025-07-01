import os, re, torch, numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import Patch                       # ← NUEVO
from pathlib import Path
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import Config
from train_test4 import CloudDeepLabV3Plus                 # cámbiala si vive en otro módulo

# ─────────────────────────── 1) PALETA Y FUNCIONES AUXILIARES ─────────────────────
PALETTE = [
    (255,255,255), (128,0,0), (0,128,0),
    (0,0,0),   (0,0,128), (128,0,128)
]
FLAT_PAL = [c for rgb in PALETTE for c in rgb]

def rgb_to_idx(rgb_arr: np.ndarray, palette: list[tuple[int,int,int]]) -> np.ndarray:
    idx = np.zeros(rgb_arr.shape[:2], dtype=np.uint8)
    for i, color in enumerate(palette):
        idx[np.all(rgb_arr == color, axis=-1)] = i
    return idx

# ─────────────────────────── 2) BÚSQUEDA DE MÁSCARA GT ────────────────────────────
def find_mask(img_path: Path) -> Path | None:
    base = re.sub(r"\.(jpg|jpeg)$", "", img_path.name, flags=re.I)
    candidates = [
        img_path.with_name(f"{base}_mask.png"),
        img_path.with_name(f"{base}.png"),
        img_path.parent.parent / "labels" / f"{base}_mask.png",
        img_path.parent.parent / "masks"  / f"{base}_mask.png",
        img_path.parent.parent / "labels" / f"{base}.png",
        img_path.parent.parent / "masks"  / f"{base}.png",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None

def load_gt(mask_path: Path | None, size: tuple[int,int]) -> Image.Image:
    if mask_path is None:
        return Image.new("RGB", size, (0,0,0))
    m = Image.open(mask_path)
    if m.mode in ("P", "L", "I"):
        idx = np.array(m, dtype=np.uint8)
    else:
        idx = rgb_to_idx(np.array(m.convert("RGB")), PALETTE)
    gt = Image.fromarray(idx, mode="P")
    gt.putpalette(FLAT_PAL)
    return gt.convert("RGB").resize(size, Image.NEAREST)

# ─────────────────────────── 3) LISTA DE IMÁGENES ────────────────────────────────
BASE_DIR = Path("Balanced/train")
image_paths = [
    BASE_DIR / "images/5-113m3_jpg.rf.1a908ea089918e172ac9b1cfbc81b590.jpg",
    BASE_DIR / "images/101_jpg.rf.2a2a92bdf083fea463b938aa1f3e6bbf.jpg",
    BASE_DIR / "images/118_jpg.rf.eceeb04c2e33998be1c3ded4e4bd0fdd.jpg",
    BASE_DIR / "images/137_jpg.rf.6980a8e200cb1d6a3471c93debb03d04.jpg",
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
for img_path in image_paths:
    original = Image.open(img_path).convert("RGB")
    gt_mask  = load_gt(find_mask(img_path), original.size)

    tensor = tfm(image=np.array(original))["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        logits = logits[0] if isinstance(logits, tuple) else logits
        pred   = torch.argmax(logits, 1).squeeze().cpu().numpy()

    pred_img = Image.fromarray(pred.astype(np.uint8), mode="P")
    pred_img.putpalette(FLAT_PAL)
    pred_rgb = pred_img.convert("RGB").resize(original.size, Image.NEAREST)

    results.append((original, gt_mask, pred_rgb))

# Figura con N filas × 3 columnas
n = len(results)
fig, axs = plt.subplots(n, 3, figsize=(15, 5*n), constrained_layout=True)
for i, (orig, gt, pred) in enumerate(results):
    row = axs[i] if n > 1 else axs
    for ax, img, title in zip(row, (orig, gt, pred),
                              ("Imagen original", "Máscara GT", "Predicción")):
        ax.imshow(img)
        ax.set_title(title, fontsize=12)
        ax.axis("off")

# ─────────────── 6) LEYENDA DE COLORES POR CLASE (CUADRADOS PEQUEÑOS) ───────────
handles = [
    Patch(facecolor=np.array(rgb)/255.0, edgecolor='black', label=f"Clase {i}")
    for i, rgb in enumerate(PALETTE)
]
fig.legend(
    handles=handles,
    loc="upper center",          # debajo de la parte superior de la figura
    bbox_to_anchor=(0.5, 1.02),
    ncol=len(PALETTE),
    frameon=False,
    fontsize=11
)

# Guardar y mostrar
plt.savefig("all_predictions_grid.png", dpi=300, bbox_inches="tight")
plt.show()

if __name__ == "__main__":
    pass
