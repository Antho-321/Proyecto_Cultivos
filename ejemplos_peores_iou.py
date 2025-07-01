"""
Encuentra, para cada clase, la imagen con el peor IoU (más bajo)
y muestra su ruta junto con el valor alcanzado.
Requiere:
  pip install albumentations==1.* pillow==10.* torch torchvision
"""

# ─────────────────────────── 1) IMPORTS ────────────────────────────────────
import os, re, torch, numpy as np
from pathlib import Path
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import Config            # ← tu módulo con IMAGE_HEIGHT / WIDTH
from train_test4 import CloudDeepLabV3Plus   # o la ruta correcta al modelo

# ─────────────────────────── 2) CONFIGURA TUS RUTAS ───────────────────────
BASE_DIR      = Path("Balanced/train")
IMAGE_DIR     = BASE_DIR / "images"
MODEL_WEIGHTS = Path("/content/drive/MyDrive/colab/cultivos_deeplab_final.pt")

# ─────────────────────────── 3) PALETA Y NOMBRES DE CLASE ─────────────────
PALETTE = [
    (255,255,255), (128,0,0), (0,128,0),
    (255,255,0),   (0,0,0), (128,0,128)
]
CLASS_NAMES = [
    "Fondo", "Lengua de vaca", "Diente de león",
    "Kikuyo", "Otro", "Papa"
]
assert len(PALETTE) == len(CLASS_NAMES), "PALETTE y CLASS_NAMES deben coincidir"

# ─────────────────────────── 4) UTILIDADES ────────────────────────────────
def rgb_to_idx(rgb_arr: np.ndarray) -> np.ndarray:
    """Convierte una máscara coloreada a índices 0-N-1 según PALETTE."""
    idx = np.zeros(rgb_arr.shape[:2], dtype=np.uint8)
    for i, color in enumerate(PALETTE):
        idx[np.all(rgb_arr == color, axis=-1)] = i
    return idx

def find_mask(img_path: Path) -> Path | None:
    """Intenta localizar la GT usando varios patrones de nombre."""
    base = re.sub(r"\.(jpg|jpeg|png)$", "", img_path.name, flags=re.I)
    candidates = [
        img_path.with_name(f"{base}_mask.png"),
        img_path.with_name(f"{base}.png"),
        img_path.parent.parent / "masks"  / f"{base}_mask.png",
        img_path.parent.parent / "labels" / f"{base}_mask.png",
        img_path.parent.parent / "masks"  / f"{base}.png",
        img_path.parent.parent / "labels" / f"{base}.png",
    ]
    return next((c for c in candidates if c.exists()), None)

def load_gt_idx(mask_path: Path) -> np.ndarray:
    """Carga la máscara GT y la devuelve como índices (sin redimensionar)."""
    m = Image.open(mask_path)
    if m.mode in ("P", "L", "I"):
        return np.array(m, dtype=np.uint8)
    return rgb_to_idx(np.array(m.convert("RGB")))

def resize_nearest(arr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Redimensiona una máscara de índices con interpolación NEAREST."""
    return np.array(Image.fromarray(arr).resize(size, Image.NEAREST))

def iou_per_class(pred: np.ndarray, gt: np.ndarray, n_cls: int) -> np.ndarray:
    """Devuelve un vector IoU (n_cls,) para cada clase; NaN si no hay unión."""
    ious = np.full(n_cls, np.nan, dtype=np.float32)
    for c in range(n_cls):
        inter = np.logical_and(pred == c, gt == c).sum()
        union = np.logical_or(pred == c, gt == c).sum()
        if union > 0:
            ious[c] = inter / union
    return ious

# ─────────────────────────── 5) MODELO Y TRANSFORMACIÓN ───────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = torch.load(MODEL_WEIGHTS, map_location=device, weights_only=False)
model.eval()

pre_tf = A.Compose([
    A.Resize(Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH),
    A.Normalize(mean=(0,0,0), std=(1,1,1), max_pixel_value=255.0),
    ToTensorV2(),
])

# ─────────────────────────── 6) BÚSQUEDA DE PEORES IoU ────────────────────
# Diccionario: clase → (peor_iou, ruta_imagen)
worst_iou = {c: (1.0, None) for c in range(len(PALETTE))}

for img_path in sorted(IMAGE_DIR.iterdir()):
    if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
        continue

    mask_path = find_mask(img_path)
    if mask_path is None:
        print(f"[Aviso] GT no encontrada → {img_path.name}")
        continue

    # --- Carga y pre-procesado ---
    image      = np.array(Image.open(img_path).convert("RGB"))
    tensor_in  = pre_tf(image=image)["image"].unsqueeze(0).to(device)
    gt_idx_org = load_gt_idx(mask_path)
    gt_idx_rs  = resize_nearest(
        gt_idx_org,
        (Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT)
    )

    # --- Predicción ---
    with torch.no_grad():
        logits = model(tensor_in)
        logits = logits[0] if isinstance(logits, (tuple, list)) else logits
        pred_idx = torch.argmax(logits, 1).squeeze().cpu().numpy()

    # --- IoU imagen completa ---
    ious = iou_per_class(pred_idx, gt_idx_rs, len(PALETTE))

    # --- Actualiza mínimos por clase ---
    for c, iou in enumerate(ious):
        if np.isnan(iou):
            continue                # no aparece la clase en GT ∴ no se evalúa
        if iou < worst_iou[c][0]:
            worst_iou[c] = (float(iou), str(img_path))

# ─────────────────────────── 7) RESULTADOS ────────────────────────────────
print("\n>> Imagen con peor IoU por clase")
for c in range(len(PALETTE)):
    score, path = worst_iou[c]
    label = CLASS_NAMES[c]
    if path is None:
        print(f"Clase {c} ({label}):  —  sin ejemplos en GT")
    else:
        print(f"Clase {c} ({label}):  IoU = {score:.4f}  |  {path}")

# También puedes acceder al diccionario `worst_iou` desde otro módulo
