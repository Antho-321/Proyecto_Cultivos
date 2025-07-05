import os
import re
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from scipy.ndimage import generic_filter
from config import Config

def find_mask(img_path: str) -> str:
    base = os.path.splitext(os.path.basename(img_path))[0]
    candidate = os.path.join(Config.VAL_MASK_DIR, base + '_mask.png')
    if os.path.isfile(candidate):
        return candidate
    raise FileNotFoundError(f"No mask found for {img_path}")

# ───────────── 1) FUNCIÓN DE FILTRO DE MAYORÍA ──────────────────────
def majority_filter(label_map: np.ndarray, window_size: int = 7) -> np.ndarray:
    def _vote(window: np.ndarray) -> int:
        vals, counts = np.unique(window, return_counts=True)
        return vals[np.argmax(counts)]
    return generic_filter(label_map, function=_vote, size=window_size, mode='nearest')

# ───────────── 2) CARGA MODELO COMO SCRIPTED MODULE ──────────────────────
device = torch.device(Config.DEVICE)

model = torch.jit.load(
    "/content/drive/MyDrive/colab/cultivos_deeplab_final.pt",
    map_location=device
)
model.eval()

# ───────────── 3) LISTAS DE ARCHIVOS DE VALIDACIÓN ──────────────────────
val_imgs = sorted([
    os.path.join(Config.VAL_IMG_DIR, f)
    for f in os.listdir(Config.VAL_IMG_DIR)
    if re.search(r'\.(jpg|png|jpeg)$', f, re.I)
])


val_masks = [find_mask(p) for p in val_imgs]

# ───────────── 4) CÁLCULO DE IoU POR CLASE ──────────────────────
def compute_iou(gt: np.ndarray, pred: np.ndarray, num_classes: int) -> float:
    ious = []
    for cls in range(num_classes):
        inter = np.logical_and(gt == cls, pred == cls).sum()
        union = np.logical_or(gt == cls, pred == cls).sum()
        if union > 0:
            ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0

# Inferimos num_classes del peso de la última capa si es accesible
state = model.state_dict()
# Ajusta esta clave al nombre real de tu última capa si es distinto
final_w_key = next(k for k in state.keys() if k.endswith('weight'))
num_classes = state[final_w_key].shape[0]

# ───────────── 5) LOOP DE EVALUACIÓN ──────────────────────
miou_list = []

for img_path, mask_path in tqdm(zip(val_imgs, val_masks), total=len(val_imgs)):
    # Carga y redimensiona imagen
    orig = Image.open(img_path).convert('RGB')
    img_resized = orig.resize((Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT), Image.NEAREST)
    img_arr = np.array(img_resized)
    tensor = (torch.from_numpy(img_arr)
                   .permute(2,0,1)
                   .unsqueeze(0)
                   .float()
                   .to(device) / 255.0)

    # Carga y redimensiona máscara GT
    gt = np.array(
        Image.open(mask_path)
             .resize((Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT), Image.NEAREST),
        dtype=np.uint8
    )

    # Predicción
    with torch.no_grad():
        out = model(tensor)
        out = out[0] if isinstance(out, (tuple, list)) else out
        pred = torch.argmax(out, dim=1).squeeze().cpu().numpy().astype(np.uint8)

    # Post-procesado con filtro de mayoría
    pred = majority_filter(pred, window_size=7)

    # Calcular IoU y acumular
    miou = compute_iou(gt, pred, num_classes)
    miou_list.append(miou)

# Resultado final
final_miou = np.mean(miou_list)
print(f"Mean IoU sobre validación: {final_miou:.4f}")
