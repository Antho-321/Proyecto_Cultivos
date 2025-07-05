# validate_with_crop.py
# ─────────────────────────────────────────────────────────────────────────
#  Validación con recorte inteligente + segunda inferencia
#  Robert Alexander Patiño Chalacan · 2025-07-05
# ─────────────────────────────────────────────────────────────────────────

import os, re, cv2, torch, numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from tqdm import tqdm
from config import Config            # ← asegúrate de tener IMAGE_WIDTH, IMAGE_HEIGHT, DEVICE, etc.

# ════════════════════════════════════════════════════════════════════════
# 1) UTILIDADES DE POSPROCESADO
# ════════════════════════════════════════════════════════════════════════
def crop_around_classes(
    image: np.ndarray,
    mask:  np.ndarray,
    classes_to_find: list[int] = [1, 2, 3, 4, 5],
    margin: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (crop_img, crop_mask) alrededor de las clases solicitadas."""
    is_class_present = np.isin(mask.squeeze(), classes_to_find)
    ys, xs = np.where(is_class_present)
    if ys.size == 0:                       # no hay píxeles de esas clases
        return image, mask
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    y0 = max(0, y_min - margin); y1 = min(mask.shape[0], y_max + margin + 1)
    x0 = max(0, x_min - margin); x1 = min(mask.shape[1], x_max + margin + 1)
    return image[y0:y1, x0:x1], mask[y0:y1, x0:x1, :]

def majority_filter_batch(preds: torch.Tensor, k: int = 7) -> torch.Tensor:
    """Filtro de mayoría 2-D sobre un batch de máscaras discretas (B,H,W)."""
    B, H, W = preds.shape
    pad = k // 2
    x = preds.unsqueeze(1).float()                # (B,1,H,W)
    patches = F.unfold(x, kernel_size=k, padding=pad)  # (B,k*k,H*W)
    vals, _ = patches.mode(dim=1)                      # (B,H*W)
    return vals.view(B, H, W).long()

def refine_with_cropping(
    model: torch.jit.ScriptModule,
    img_tensor: torch.Tensor,                   # (C,H,W), rango 0-1
    classes_to_find: list[int] = [1, 2, 3, 4, 5],
    margin: int = 10
) -> torch.Tensor:                              # (H,W) Long
    """Recorte + segunda inferencia para refinar las clases objetivo."""
    device = next(model.parameters()).device

    # 1) Predicción “rápida” completa
    with torch.no_grad(), autocast(device_type=Config.DEVICE, dtype=torch.float16):
        logits = model(img_tensor.unsqueeze(0).to(device))
        logits = logits[0] if isinstance(logits, (tuple, list)) else logits
        pred   = logits.argmax(1).squeeze(0).cpu()          # (H,W)

    # 2) Recorte alrededor de las clases solicitadas
    img_np  = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    mask_np = pred.numpy()[..., None]
    crop_img, crop_mask = crop_around_classes(img_np, mask_np, classes_to_find, margin)

    # 2 bis) Si no hay clases => se devuelve la predicción original
    if crop_img is img_np:
        return pred

    # 3) Re-escalado del recorte y segunda inferencia
    crop_resized = cv2.resize(
        crop_img,
        (Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT),
        interpolation=cv2.INTER_NEAREST
    )
    crop_tensor = torch.from_numpy(crop_resized.astype(np.float32) / 255.0) \
                       .permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad(), autocast(device_type=Config.DEVICE, dtype=torch.float16):
        logits_crop = model(crop_tensor)
        logits_crop = logits_crop[0] if isinstance(logits_crop, (tuple, list)) else logits_crop
        refined     = logits_crop.argmax(1).squeeze(0).cpu().numpy()

    # 4) Devuelve la máscara refinada pegada en su lugar original
    refined = cv2.resize(
        refined.astype(np.uint8),
        (crop_mask.shape[1], crop_mask.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )
    ys, xs = np.where(np.isin(mask_np.squeeze(), classes_to_find))
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    y0 = max(0, y_min - margin); y1 = min(mask_np.shape[0], y_max + margin + 1)
    x0 = max(0, x_min - margin); x1 = min(mask_np.shape[1], x_max + margin + 1)
    final_pred = pred.numpy()
    final_pred[y0:y1, x0:x1] = refined
    return torch.from_numpy(final_pred)

# ════════════════════════════════════════════════════════════════════════
# 2) DATASET & DATALOADER
# ════════════════════════════════════════════════════════════════════════
class ValDataset(Dataset):
    """Carga imágenes RGB y máscaras单-canal, las reescala y normaliza."""
    def __init__(self, img_paths, mask_paths):
        self.imgs, self.masks = img_paths, mask_paths
    def __len__(self): return len(self.imgs)
    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.imgs[idx], cv2.IMREAD_COLOR),
                           cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks[idx], cv2.IMREAD_UNCHANGED)
        if mask.ndim == 3: mask = mask[..., 0]
        size = (Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT)
        img  = cv2.resize(img,  size, interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
        img_tensor  = torch.from_numpy(img.astype(np.float32)/255.0).permute(2,0,1)
        mask_tensor = torch.from_numpy(mask.astype(np.int64))
        return img_tensor, mask_tensor

# Rutas de validación
val_imgs = sorted([
    os.path.join(Config.VAL_IMG_DIR, f)
    for f in os.listdir(Config.VAL_IMG_DIR)
    if re.search(r'\.(jpg|png|jpeg)$', f, re.I)
])
val_masks = [
    os.path.join(Config.VAL_MASK_DIR,
                 os.path.splitext(os.path.basename(p))[0] + '_mask.png')
    for p in val_imgs
]
val_loader = DataLoader(
    ValDataset(val_imgs, val_masks),
    batch_size=8, num_workers=4, pin_memory=True, shuffle=False
)

# ════════════════════════════════════════════════════════════════════════
# 3) MODELO
# ════════════════════════════════════════════════════════════════════════
device = torch.device(Config.DEVICE)
model = torch.jit.load(
    "/content/drive/MyDrive/colab/cultivos_deeplab_final.pt",
    map_location=device
).eval()

# Número de clases a partir del último conv
state = model.state_dict()
final_w_key = next(k for k in state if k.endswith("weight"))
num_classes = state[final_w_key].shape[0]

# ════════════════════════════════════════════════════════════════════════
# 4) VALIDACIÓN
# ════════════════════════════════════════════════════════════════════════
conf_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)

with torch.no_grad():
    for imgs, gts in tqdm(val_loader, desc="Validación"):
        # muevo a CPU para la función de refino (opcional)
        gts = gts.to(device, non_blocking=True)
        refined_preds = []
        for b in range(imgs.size(0)):
            pred_b = refine_with_cropping(
                model,
                imgs[b],                     # tensor (C,H,W) en CPU
                classes_to_find=[1,2,3,4,5],
                margin=10
            )
            refined_preds.append(pred_b)
        preds = torch.stack(refined_preds).to(device)

        # Filtro de mayoría para suavizar errores puntuales
        preds = majority_filter_batch(preds, k=7)

        # Matriz de confusión
        idx = num_classes * gts.view(-1) + preds.view(-1)
        conf_matrix += torch.bincount(idx, minlength=num_classes**2) \
                           .view(num_classes, num_classes)

# ════════════════════════════════════════════════════════════════════════
# 5) Cálculo de mIoU
# ════════════════════════════════════════════════════════════════════════
diag  = conf_matrix.diag().float()
union = conf_matrix.sum(dim=0) + conf_matrix.sum(dim=1) - diag
valid = union > 0
iou_per_class = diag[valid] / union[valid]
mean_iou = iou_per_class.mean().item()
print(f"Mean IoU sobre validación: {mean_iou:.4f}")
