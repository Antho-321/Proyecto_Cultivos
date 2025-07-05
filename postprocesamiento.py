import os
import re
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
from tqdm import tqdm

from config import Config

# ───────────── 1) DATASET & DATALOADER ──────────────────────
class ValDataset(Dataset):
    def __init__(self, img_paths, mask_paths):
        self.imgs  = img_paths
        self.masks = mask_paths

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # carga imagen BGR → RGB
        img = cv2.imread(self.imgs[idx], cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # carga máscara en escala de grises
        mask = cv2.imread(self.masks[idx], cv2.IMREAD_UNCHANGED)
        if mask.ndim == 3:
            mask = mask[..., 0]

        # resize ambos
        size = (Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT)
        img  = cv2.resize(img,  size, interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)

        # normalizar y convertir a tensor
        img_tensor  = torch.from_numpy(img.astype(np.float32) / 255.0) \
                           .permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask.astype(np.int64))

        return img_tensor, mask_tensor

val_imgs = sorted([
    os.path.join(Config.VAL_IMG_DIR, f)
    for f in os.listdir(Config.VAL_IMG_DIR)
    if re.search(r'\.(jpg|png|jpeg)$', f, re.I)
])
val_masks = [
    os.path.join(
        Config.VAL_MASK_DIR,
        os.path.splitext(os.path.basename(p))[0] + '_mask.png'
    )
    for p in val_imgs
]

val_loader = DataLoader(
    ValDataset(val_imgs, val_masks),
    batch_size=8,               # ajusta según tu GPU
    num_workers=4,              # paraleliza I/O
    pin_memory=True,
    shuffle=False,
)

# ───────────── 2) MAJORITY FILTER EN GPU ──────────────────────
def majority_filter_batch(preds: torch.Tensor, k: int = 7) -> torch.Tensor:
    # preds: (B, H, W) integer class indices
    B, H, W = preds.shape
    pad = k // 2
    x = preds.unsqueeze(1).float()  # (B,1,H,W)
    patches = F.unfold(x, kernel_size=k, padding=pad)           # (B, k*k, H*W)
    vals, _ = patches.mode(dim=1)                               # (B, H*W)
    return vals.view(B, H, W).long()

# ───────────── 3) CARGA SCRIPTED MODEL ──────────────────────
device = torch.device(Config.DEVICE)
model = torch.jit.load(
    "/content/drive/MyDrive/colab/cultivos_deeplab_final.pt",
    map_location=device
)
model.eval()

# inferir número de clases
state = model.state_dict()
final_w_key = next(k for k in state.keys() if k.endswith('weight'))
num_classes = state[final_w_key].shape[0]

# ───────────── 4) EVALUACIÓN Y mIoU ──────────────────────
conf_matrix = torch.zeros(
    (num_classes, num_classes),
    dtype=torch.int64,
    device=device
)

with torch.no_grad():
    for imgs, gts in tqdm(val_loader, desc="Validación"):
        imgs = imgs.to(device, non_blocking=True)
        gts  = gts.to(device, non_blocking=True)

        # inferencia en FP16
        with autocast():
            out = model(imgs)
            out = out[0] if isinstance(out, (tuple, list)) else out
            preds = out.argmax(dim=1)

        # post-procesado: filtro de mayoría
        preds = majority_filter_batch(preds, k=7)

        # acumular matriz de confusión
        idx = num_classes * gts.view(-1) + preds.view(-1)
        cm = torch.bincount(
            idx,
            minlength=num_classes**2
        ).view(num_classes, num_classes)
        conf_matrix += cm

# calcular IoU por clase y mean IoU
diag = conf_matrix.diag().float()
union = (
    conf_matrix.sum(dim=0)
  + conf_matrix.sum(dim=1)
  - diag
)
iou_per_class = diag / union
mean_iou = iou_per_class.mean().item()

print(f"Mean IoU sobre validación: {mean_iou:.4f}")
