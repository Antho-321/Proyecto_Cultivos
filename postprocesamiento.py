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

def crop_around_classes(
    image: np.ndarray,
    mask: np.ndarray,
    classes_to_find: list[int] = [1, 2, 3, 4, 5],
    margin: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """
    Recorta un rectángulo alrededor de todos los píxeles que pertenecen
    a las clases especificadas en la máscara.
    """
    is_class_present = np.isin(mask.squeeze(), classes_to_find)
    ys, xs = np.where(is_class_present)
    if ys.size == 0:
        return image, mask
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    y0 = max(0, y_min - margin)
    y1 = min(mask.shape[0], y_max + margin + 1)
    x0 = max(0, x_min - margin)
    x1 = min(mask.shape[1], x_max + margin + 1)
    cropped_image = image[y0:y1, x0:x1]
    cropped_mask = mask[y0:y1, x0:x1, :]
    return cropped_image, cropped_mask

class ValDataset(Dataset):
    def __init__(self, img_paths, mask_paths,
                 classes_to_find=[1,2,3,4,5], margin=10):
        self.imgs = img_paths
        self.masks = mask_paths
        self.classes_to_find = classes_to_find
        self.margin = margin

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = cv2.imread(self.imgs[idx], cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks[idx], cv2.IMREAD_UNCHANGED)
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask_3d = mask[..., np.newaxis]
        img_crop, mask_crop = crop_around_classes(
            img, mask_3d,
            classes_to_find=self.classes_to_find,
            margin=self.margin
        )
        mask_crop = mask_crop[..., 0]
        size = (Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT)
        img_resized  = cv2.resize(img_crop,  size, interpolation=cv2.INTER_NEAREST)
        mask_resized = cv2.resize(mask_crop, size, interpolation=cv2.INTER_NEAREST)
        img_tensor  = torch.from_numpy(img_resized.astype(np.float32) / 255.0) \
                           .permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask_resized.astype(np.int64))
        return img_tensor, mask_tensor

def majority_filter_batch(preds: torch.Tensor, k: int = 7) -> torch.Tensor:
    B, H, W = preds.shape
    pad = k // 2
    x = preds.unsqueeze(1).float()
    patches = F.unfold(x, kernel_size=k, padding=pad)
    vals, _ = patches.mode(dim=1)
    return vals.view(B, H, W).long()

def evaluate():
    # 1) rutas de validación
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

    # 2) DataLoader
    val_loader = DataLoader(
        ValDataset(val_imgs, val_masks,
                   classes_to_find=[1,2,3,4,5],
                   margin=10),
        batch_size=8,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
    )

    # 3) modelo
    device = torch.device(Config.DEVICE)
    model = torch.jit.load(
        "/content/drive/MyDrive/colab/cultivos_deeplab_final.pt",
        map_location=device
    )
    model.eval()
    state = model.state_dict()
    final_w_key = next(k for k in state.keys() if k.endswith('weight'))
    num_classes = state[final_w_key].shape[0]

    # 4) evaluación
    conf_matrix = torch.zeros(
        (num_classes, num_classes),
        dtype=torch.int64,
        device=device
    )

    with torch.no_grad():
        for imgs, gts in tqdm(val_loader, desc="Validación"):
            imgs = imgs.to(device, non_blocking=True)
            gts  = gts.to(device, non_blocking=True)
            with autocast(device_type=Config.DEVICE, dtype=torch.float16):
                out = model(imgs)
                out = out[0] if isinstance(out, (tuple, list)) else out
                preds = out.argmax(dim=1)
            preds = majority_filter_batch(preds, k=7)
            idx = num_classes * gts.view(-1) + preds.view(-1)
            cm = torch.bincount(
                idx,
                minlength=num_classes**2
            ).view(num_classes, num_classes)
            conf_matrix += cm

    diag  = conf_matrix.diag().float()
    union = conf_matrix.sum(dim=0) + conf_matrix.sum(dim=1) - diag
    valid = union > 0
    iou_per_class = diag[valid] / union[valid]
    mean_iou = iou_per_class.mean().item()
    print(f"Mean IoU sobre validación: {mean_iou:.4f}")

if __name__ == "__main__":
    evaluate()