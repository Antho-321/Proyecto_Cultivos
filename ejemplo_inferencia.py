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
    (255,255,0),   (0,0,0), (128,0,128)
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
BASE_DIR = Path("Balanced/val")
image_paths = [
    BASE_DIR / "images/out_focus_5-105m3_jpg.rf.6aa735c9180337fbcc3e0b778038e7f9.jpg",
    BASE_DIR / "images/rotate90_DJI_0058-JPG_2250_250_JPG.rf.8aa8dbebf43c94149e0ca7ddd4c9cf97.jpg",
    BASE_DIR / "images/rotate90_5-318m3_jpg.rf.2c05aba51bc931e14faef3ce46fcb30c.jpg",
    BASE_DIR / "images/rotate90_DJI_0058-JPG_2250_250_JPG.rf.8aa8dbebf43c94149e0ca7ddd4c9cf97.jpg",
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

# 0 ───── LISTA DE NOMBRES DE CLASE  ───────────────────────────────────────────
# Ajusta estos nombres al orden exacto de tu PALETTE
CLASS_NAMES = [
    "Fondo",   # índice 0  → (255,255,255)
    "Lengua de vaca",                # índice 1  → (128,0,0)
    "Diente de león",          # índice 2  → (0,128,0)
    "Kikuyo",        # índice 3  → (255,255,0)
    "Otro",       # índice 4  → (0,0,0)
    "Papa",    # índice 5  → (128,0,128)
]
# Comprueba que len(CLASS_NAMES) == len(PALETTE)
assert len(CLASS_NAMES) == len(PALETTE), "‽ Mismos elementos en PALETTE y CLASS_NAMES"

# … (todo tu código previo sin cambios) …

# 6 ───── LEYENDA DE COLORES POR CLASE (CUADRADOS PEQUEÑOS)  ──────────────────
handles = [
    Patch(facecolor=np.array(rgb)/255.0,
          edgecolor="black",
          label=CLASS_NAMES[i])          # ← ahora usa el nombre descriptivo
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

# Guardar y mostrar
plt.savefig("all_predictions_grid.png", dpi=300, bbox_inches="tight")
plt.show()

if __name__ == "__main__":
    pass
