# train_model2.py  –  Tesla T4-tuned
import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ─── Global perf switches ──────────────────────────────────────────────────────
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

torch.set_flush_denormal(True)
torch.set_num_threads(4)
torch.set_num_interop_threads(2)
torch.backends.cudnn.benchmark = True         # keeps fastest conv algo
torch.backends.cudnn.workspace_limit = 512 * 1024 * 1024   # 512 MB

# ─── Tu arquitectura y utilidades ──────────────────────────────────────────────
from model2 import CloudDeepLabV3Plus
from utils import (
    imprimir_distribucion_clases_post_augmentation,
    crop_around_classes,
    save_performance_plot
)
from config import Config

# =================================================================================
# 1. DATASET PERSONALIZADO
# =================================================================================
class CloudDataset(torch.utils.data.Dataset):
    _IMG_EXTENSIONS = ('.jpg', '.png')

    def __init__(self, image_dir: str, mask_dir: str, transform: A.Compose | None = None):
        self.image_dir = image_dir
        self.mask_dir  = mask_dir
        self.transform = transform
        self.images    = [f for f in os.listdir(image_dir) if f.lower().endswith(self._IMG_EXTENSIONS)]

    def __len__(self):
        return len(self.images)

    def _mask_path_from_image_name(self, image_filename: str):
        name_without_ext = image_filename.rsplit('.', 1)[0]
        return os.path.join(self.mask_dir, f"{name_without_ext}_mask.png")

    def __getitem__(self, idx: int):
        img_filename = self.images[idx]
        img_path     = os.path.join(self.image_dir, img_filename)
        mask_path    = self._mask_path_from_image_name(img_filename)

        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Máscara no encontrada para {img_filename} en {mask_path}")

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)[:, :, ::-1]   # BGR → RGB
        mask  = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        mask_3d                  = np.expand_dims(mask, axis=-1)
        image_cropped, mask_3d_c = crop_around_classes(image, mask_3d)
        mask_cropped             = mask_3d_c.squeeze()

        if self.transform:
            augmented = self.transform(image=image_cropped, mask=mask_cropped)
            image     = augmented["image"].unsqueeze(0).contiguous(memory_format=torch.channels_last).squeeze(0)
            mask      = augmented["mask"].contiguous()

        return image, mask

# =================================================================================
# 2. ENTRENAMIENTO, VALIDACIÓN Y MÉTRICAS
# =================================================================================
def train_fn(loader, model, optimizer, loss_fn, scaler, num_classes=6):
    loop = tqdm(loader, leave=True)
    model.train()

    tp = torch.zeros(num_classes, device=Config.DEVICE)
    fp = torch.zeros(num_classes, device=Config.DEVICE)
    fn = torch.zeros(num_classes, device=Config.DEVICE)

    for data, targets in loop:
        data    = data.to(Config.DEVICE, non_blocking=True, memory_format=torch.channels_last).half()
        targets = targets.to(Config.DEVICE, non_blocking=True).long()

        with autocast(device_type="cuda", dtype=torch.float16):
            preds = model(data)
            preds = preds[0] if isinstance(preds, tuple) else preds
            loss  = loss_fn(preds, targets)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        _, pred_cls = torch.max(preds, dim=1)
        for c in range(num_classes):
            tp[c] += ((pred_cls == c) & (targets == c)).sum()
            fp[c] += ((pred_cls == c) & (targets != c)).sum()
            fn[c] += ((pred_cls != c) & (targets == c)).sum()

        loop.set_postfix(loss=loss.item())

    eps = 1e-6
    iou_per_class  = tp / (tp + fp + fn + eps)
    dice_per_class = (2 * tp) / (2 * tp + fp + fn + eps)
    mean_iou       = torch.nanmean(iou_per_class)

    print("\nFin de la época:")
    print(f"  Dice por clase: {dice_per_class.cpu().numpy()}")
    print(f"  IoU  por clase: {iou_per_class.cpu().numpy()}")
    print(f"  mIoU: {mean_iou:.4f}")

    return mean_iou

def check_metrics(loader, model, n_classes=6, device="cuda"):
    eps = 1e-8
    conf_mat = torch.zeros((n_classes, n_classes), dtype=torch.float64, device=device)
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True).half(), y.to(device, non_blocking=True).long()
            logits = model(x)
            logits = logits[0] if isinstance(logits, tuple) else logits
            preds  = torch.argmax(logits, dim=1)

            flat = (preds * n_classes + y).view(-1).float()
            conf = torch.histc(flat, bins=n_classes*n_classes, min=0, max=n_classes*n_classes-1).view(n_classes, n_classes)
            conf_mat += conf

    inter = torch.diag(conf_mat)
    pred_sum, true_sum = conf_mat.sum(dim=1), conf_mat.sum(dim=0)
    union = pred_sum + true_sum - inter

    iou  = (inter+eps) / (union+eps)
    dice = (2*inter+eps) / (pred_sum+true_sum+eps)

    model.train()
    print("IoU :", iou.cpu().numpy())
    print("Dice:", dice.cpu().numpy())
    print("mIoU:", iou.mean().item())
    return iou.mean(), dice.mean()

def validate_fn(loader, model, loss_fn, device=Config.DEVICE):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True).half(), y.to(device, non_blocking=True).long()
            logits = model(x)
            logits = logits[0] if isinstance(logits, tuple) else logits
            total += loss_fn(logits, y).item() * x.size(0)
    model.train()
    return total / len(loader.dataset)

# =================================================================================
# 3. FUNCIÓN PRINCIPAL
# =================================================================================
def main():

    print(f"Dispositivo: {Config.DEVICE}")

    train_tf = A.Compose([
        A.Resize(Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH),
        A.Rotate(limit=35, p=0.7),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Normalize([0, 0, 0], [1, 1, 1], max_pixel_value=255.0),
        ToTensorV2(),
    ])
    val_tf = A.Compose([
        A.Resize(Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH),
        A.Normalize([0, 0, 0], [1, 1, 1], max_pixel_value=255.0),
        ToTensorV2(),
    ])

    train_ds = CloudDataset(Config.TRAIN_IMG_DIR, Config.TRAIN_MASK_DIR, train_tf)
    val_ds   = CloudDataset(Config.VAL_IMG_DIR,   Config.VAL_MASK_DIR,   val_tf)

    # ─── DataLoader optimizado ─────────────────────────────────────────────────
    train_ld = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        pin_memory_device="cuda",
        persistent_workers=True,
        prefetch_factor=4,
    )

    val_ld = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        pin_memory_device="cuda",
        persistent_workers=True,
        prefetch_factor=4,
    )

    imprimir_distribucion_clases_post_augmentation(
        train_ld, 6, "Distribución de clases en ENTRENAMIENTO (post-aug)"
    )

    # ─── Modelo ───────────────────────────────────────────────────────────────
    model = CloudDeepLabV3Plus(num_classes=6).to(
        Config.DEVICE,
        memory_format=torch.channels_last
    )

    print("Compilando modelo…")
    model = torch.compile(model, backend="inductor", mode="default", fullgraph=False)

    loss_fn   = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        fused=True,
        capturable=True
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.3,
        patience=20,
        min_lr=1e-7,
        verbose=True
    )
    scaler = GradScaler()

    best_miou = -1.0
    tr_hist, val_hist = [], []

    for epoch in range(Config.NUM_EPOCHS):
        print(f"\n— Epoch {epoch + 1}/{Config.NUM_EPOCHS} —")
        tr_miou  = train_fn(train_ld, model, optimizer, loss_fn, scaler)
        val_loss = validate_fn(val_ld, model, loss_fn)
        cur_miou, cur_dice = check_metrics(val_ld, model, 6, Config.DEVICE)
        scheduler.step(val_loss)

        tr_hist.append(tr_miou.item())
        val_hist.append(cur_miou.item())

        if cur_miou > best_miou:
            best_miou = cur_miou
            print(f"🔹 Nuevo mejor mIoU: {best_miou:.4f} | Dice: {cur_dice:.4f} → guardando…")
            torch.save({
                "epoch":     epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_mIoU": best_miou,
            }, Config.MODEL_SAVE_PATH)

    save_performance_plot(
        tr_hist, val_hist,
        "/content/drive/MyDrive/colab/rendimiento_miou.png"
    )

    print("\nEvaluando mejor modelo guardado…")
    ckpt = torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    best_miou, best_dice = check_metrics(val_ld, model, 6, Config.DEVICE)
    print(f"mIoU final: {best_miou:.4f} | Dice final: {best_dice:.4f}")


if __name__ == "__main__":
    main()