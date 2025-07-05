import os
import re
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from scipy.ndimage import generic_filter
from config import Config
from model2 import CloudDeepLabV3Plus  # Ajusta el import a donde definas tu clase

# ───────────── 1) FUNCIÓN DE FILTRO DE MAYORÍA ──────────────────────
def majority_filter(label_map: np.ndarray, window_size: int = 7) -> np.ndarray:
    def _vote(window: np.ndarray) -> int:
        vals, counts = np.unique(window, return_counts=True)
        return vals[np.argmax(counts)]
    return generic_filter(label_map, function=_vote, size=window_size, mode='nearest')

# ───────────── 2) CARGA MODELO ──────────────────────
device = torch.device(Config.DEVICE)

# 1) Instantiate
model = CloudDeepLabV3Plus(num_classes=6)  # reemplaza 6 por tu número real de clases

# 2) Load weights
checkpoint = torch.load(
    "/content/drive/MyDrive/colab/cultivos_deeplab_final.pt",
    map_location=device,
    weights_only=False    # ← explicit override
)
# Si guardaste sólo state_dict:
model.load_state_dict(checkpoint['state_dict'])
# Si guardaste todo el modelo directamente:
# model = checkpoint
model.to(device).eval()

# ───────────── 3) LISTAS DE ARCHIVOS ──────────────────────
val_imgs  = sorted([
    os.path.join(Config.VAL_IMG_DIR, f)
    for f in os.listdir(Config.VAL_IMG_DIR)
    if re.search(r'\.(jpg|png|jpeg)$', f, re.I)
])
val_masks = [
    os.path.join(
        Config.VAL_MASK_DIR,
        os.path.splitext(os.path.basename(p))[0] + '.png'
    )
    for p in val_imgs
]

# ───────────── 4) MÉTRICA IoU ──────────────────────
def compute_iou(gt: np.ndarray, pred: np.ndarray, num_classes: int) -> float:
    ious = []
    for cls in range(num_classes):
        inter = np.logical_and(gt == cls, pred == cls).sum()
        union = np.logical_or(gt == cls, pred == cls).sum()
        if union > 0:
            ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0

# ───────────── 5) EVALUACIÓN ──────────────────────
miou_list = []
num_classes = model.classifier[-1].out_channels \
    if hasattr(model, 'classifier') else 6  # o el atributo que defina tus clases

for img_path, mask_path in tqdm(zip(val_imgs, val_masks), total=len(val_imgs)):
    # carga imagen y GT
    orig = Image.open(img_path).convert('RGB')
    gt = np.array(
        Image.open(mask_path)
             .resize((Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT), Image.NEAREST),
        dtype=np.uint8
    )

    # preprocess
    img = np.array(
        orig.resize((Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT), Image.NEAREST)
    )
    tensor = (
        torch.from_numpy(img).permute(2,0,1)
             .unsqueeze(0).float().to(device) / 255.0
    )

    with torch.no_grad():
        out = model(tensor)
        out = out[0] if isinstance(out, (tuple,list)) else out
        pred = torch.argmax(out, dim=1).squeeze().cpu().numpy().astype(np.uint8)

    # post-process con majority filter
    pred = majority_filter(pred, window_size=7)

    # calcular IoU
    miou = compute_iou(gt, pred, num_classes)
    miou_list.append(miou)

final_miou = np.mean(miou_list)
print(f"Mean IoU sobre validación: {final_miou:.4f}")
