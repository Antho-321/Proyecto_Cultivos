import os
import re
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from scipy.ndimage import generic_filter
from config import Config
from model import CloudDeepLabV3Plus  # o donde tengas tu clase

# ───────────── 1) FUNCIÓN DE FILTRO DE MAYORÍA ──────────────────────
def majority_filter(label_map: np.ndarray, window_size: int = 7) -> np.ndarray:
    def _vote(window: np.ndarray) -> int:
        vals, counts = np.unique(window, return_counts=True)
        return vals[np.argmax(counts)]
    return generic_filter(label_map, function=_vote, size=window_size, mode='nearest')

# ───────────── 2) CARGA MODELO ──────────────────────
device = torch.device(Config.DEVICE)
model = torch.load(Config.MODEL_SAVE_PATH, map_location=device, weights_only=False)
model.eval()

# ───────────── 3) LISTAS DE ARCHIVOS ──────────────────────
val_imgs  = sorted([os.path.join(Config.VAL_IMG_DIR,  f)
                    for f in os.listdir(Config.VAL_IMG_DIR)
                    if re.search(r'\.(jpg|png|jpeg)$', f, re.I)])
val_masks = [os.path.join(Config.VAL_MASK_DIR,
                    os.path.splitext(os.path.basename(p))[0] + '.png')
             for p in val_imgs]

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
num_classes = len(model.state_dict()['classifier.4.weight']) \
    if 'classifier.4.weight' in model.state_dict() else model.num_classes

for img_path, mask_path in tqdm(zip(val_imgs, val_masks), total=len(val_imgs)):
    # carga imagen
    orig = Image.open(img_path).convert('RGB')
    w,h = orig.size

    # carga máscara GT (asume PNG con índices de clase)
    gt = np.array(Image.open(mask_path).resize((Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT), Image.NEAREST), dtype=np.uint8)

    # preprocesado (igual que en training)
    img = np.array(orig.resize((Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT), Image.NEAREST))
    tensor = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float().to(device) / 255.0
    # (añade aquí Normalize si la usas en tfm)

    with torch.no_grad():
        out = model(tensor)
        out = out[0] if isinstance(out, (tuple,list)) else out
        pred = torch.argmax(out, dim=1).squeeze().cpu().numpy().astype(np.uint8)

    # filtro de mayoría 7×7
    pred = majority_filter(pred, window_size=7)

    # calcula IoU para esta imagen
    miou = compute_iou(gt, pred, num_classes)
    miou_list.append(miou)

# mean IoU final
final_miou = np.mean(miou_list)
print(f"Mean IoU sobre validación: {final_miou:.4f}")
