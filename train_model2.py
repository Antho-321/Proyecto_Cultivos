# train_model2.py

import os
# ─── CPU-side thread tuning ─────────────────────────────────────────────────────
os.environ["OMP_NUM_THREADS"]   = "4"
os.environ["MKL_NUM_THREADS"]   = "4"

import torch
# pin PyTorch threads to avoid oversubscription
torch.set_num_threads(4)
torch.set_num_interop_threads(2)

import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.checkpoint import checkpoint_sequential

# ─── Enable TF32 on Ampere+ GPUs ────────────────────────────────────────────────
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark      = True
torch.set_float32_matmul_precision("high")

# tu arquitectura y utilidades
from model2 import CloudDeepLabV3Plus
from utils import (
    imprimir_distribucion_clases_post_augmentation,
    crop_around_classes,
    save_performance_plot
)
from config import Config

# =================================================================================
# 1. WRAPPER PARA CHECKPOINT SEQUENTIAL
# =================================================================================
class CheckpointedCloudDeepLabV3Plus(nn.Module):
    def __init__(self, base_model: nn.Module, segments: int):
        super().__init__()
        # descompone el modelo en sub-módulos ordenados
        self.functions = list(base_model.children())
        self.segments  = segments

    def forward(self, x):
        # pasa la entrada a través de los segmentos con checkpointing
        return checkpoint_sequential(self.functions, self.segments, x)

# =================================================================================
# 2. DATASET PERSONALIZADO (MODIFICADO)
# =================================================================================
class CloudDataset(torch.utils.data.Dataset):
    _IMG_EXTENSIONS = ('.jpg', '.png')

    def __init__(self, image_dir: str, mask_dir: str, transform: A.Compose | None = None):
        self.image_dir = image_dir
        self.mask_dir  = mask_dir
        self.transform = transform
        self.images    = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith(self._IMG_EXTENSIONS)
        ]

    def __len__(self) -> int:
        return len(self.images)

    def _mask_path_from_image_name(self, image_filename: str) -> str:
        name_without_ext = image_filename.rsplit('.', 1)[0]
        mask_filename    = f"{name_without_ext}_mask.png"
        return os.path.join(self.mask_dir, mask_filename)

    def __getitem__(self, idx: int):
        img_filename = self.images[idx]
        img_path     = os.path.join(self.image_dir, img_filename)
        mask_path    = self._mask_path_from_image_name(img_filename)
        
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Máscara no encontrada para {img_filename} en {mask_path}")

        image = np.array(Image.open(img_path).convert("RGB"))
        mask  = np.array(Image.open(mask_path).convert("L"))

        # --- Recorte ANTES de las transformaciones ---
        mask_3d      = np.expand_dims(mask, axis=-1)
        image_cropped, mask_cropped_3d = crop_around_classes(image, mask_3d)
        mask_cropped = mask_cropped_3d.squeeze()

        if self.transform:
            augmented = self.transform(image=image_cropped, mask=mask_cropped)
            image     = augmented["image"]
            mask      = augmented["mask"]

        return image, mask

# =================================================================================
# 3. FUNCIONES DE ENTRENAMIENTO Y VALIDACIÓN
# =================================================================================
def train_fn(loader, model, optimizer, loss_fn, scaler, num_classes=6):
    loop = tqdm(loader, leave=True)
    model.train()

    tp = torch.zeros(num_classes, device=Config.DEVICE)
    fp = torch.zeros(num_classes, device=Config.DEVICE)
    fn = torch.zeros(num_classes, device=Config.DEVICE)

    for data, targets in loop:
        # ─── move to GPU as NHWC float16 ────────────────────────────────────────
        data    = data.to(
            Config.DEVICE,
            non_blocking=True,
            memory_format=torch.channels_last
        )
        targets = targets.to(Config.DEVICE, non_blocking=True).long()

        with autocast(device_type=Config.DEVICE, dtype=torch.float16):
            output      = model(data)
            predictions = output[0] if isinstance(output, tuple) else output
            loss        = loss_fn(predictions, targets)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        _, predicted_classes = torch.max(predictions, dim=1)

        for c in range(num_classes):
            tp[c] += ((predicted_classes == c) & (targets == c)).sum()
            fp[c] += ((predicted_classes == c) & (targets != c)).sum()
            fn[c] += ((predicted_classes != c) & (targets == c)).sum()

        loop.set_postfix(loss=loss.item())

    eps = 1e-6
    iou_per_class  = tp / (tp + fp + fn + eps)
    dice_per_class = (2 * tp) / (2 * tp + fp + fn + eps)
    mean_iou       = torch.nanmean(iou_per_class)

    print("\nÉpoca de entrenamiento finalizada:")
    print(f"  - Dice por clase: {dice_per_class.cpu().numpy()}")
    print(f"  - IoU  por clase: {iou_per_class.cpu().numpy()}")
    print(f"  - mIoU: {mean_iou:.4f}")

    return mean_iou

def check_metrics(loader, model, n_classes: int = 6, device: str = "cuda"):
    eps      = 1e-8
    conf_mat = torch.zeros((n_classes, n_classes), dtype=torch.float64, device=device)

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).long()

            logits = model(x)
            logits = logits[0] if isinstance(logits, tuple) else logits
            preds  = torch.argmax(logits, dim=1)

            flat = (preds * n_classes + y).view(-1).float()
            conf = torch.histc(
                flat,
                bins=n_classes * n_classes,
                min=0,
                max=n_classes * n_classes - 1
            ).view(n_classes, n_classes)
            conf_mat += conf

    intersection     = torch.diag(conf_mat)
    pred_sum         = conf_mat.sum(dim=1)
    true_sum         = conf_mat.sum(dim=0)
    union            = pred_sum + true_sum - intersection

    iou_per_class    = (intersection + eps) / (union + eps)
    dice_per_class   = (2 * intersection + eps) / (pred_sum + true_sum + eps)
    miou_macro       = iou_per_class.mean()
    dice_macro       = dice_per_class.mean()

    print("IoU por clase :", iou_per_class.cpu().numpy())
    print("Dice por clase:", dice_per_class.cpu().numpy())
    print(f"mIoU macro = {miou_macro:.4f} | Dice macro = {dice_macro:.4f}")

    model.train()
    return miou_macro, dice_macro

def validate_fn(loader, model, loss_fn, device=Config.DEVICE):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).long()

            logits = model(x)
            logits = logits[0] if isinstance(logits, tuple) else logits
            loss   = loss_fn(logits, y)
            total_loss += loss.item() * x.size(0)

    model.train()
    return total_loss / len(loader.dataset)

# =================================================================================
# 4. FUNCIÓN PRINCIPAL DE EJECUCIÓN
# =================================================================================
def main():
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(0.95)

    print(f"Using device: {Config.DEVICE}")

    train_transform = A.Compose([
        A.Resize(Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH),
        A.Rotate(limit=35, p=0.7),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Normalize([0,0,0], [1,1,1], max_pixel_value=255.0),
        ToTensorV2(),
    ])
    val_transform = A.Compose([
        A.Resize(Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH),
        A.Normalize([0,0,0], [1,1,1], max_pixel_value=255.0),
        ToTensorV2(),
    ])

    train_dataset = CloudDataset(
        image_dir=Config.TRAIN_IMG_DIR,
        mask_dir=Config.TRAIN_MASK_DIR,
        transform=train_transform
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        shuffle=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    val_dataset = CloudDataset(
        image_dir=Config.VAL_IMG_DIR,
        mask_dir=Config.VAL_MASK_DIR,
        transform=val_transform
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        shuffle=False
    )

    imprimir_distribucion_clases_post_augmentation(
        train_loader, 6,
        "Distribución de clases en ENTRENAMIENTO (post-aug)"
    )

    # ─── Modelo base
    base_model = CloudDeepLabV3Plus(num_classes=6)
    # ─── Wrap con checkpointing
    model = CheckpointedCloudDeepLabV3Plus(
        base_model=base_model,
        segments=4
    ).to(Config.DEVICE)

    # ─── NHWC + full float16
    model = model.to(
        Config.DEVICE,
        memory_format=torch.channels_last
    ).half()

    print("Compiling the model… (this may take a minute)")
    torch._inductor.config.triton.unique_kernel_names = True
    torch._inductor.config.epilogue_fusion           = "max"
    torch._inductor.config.triton.cudagraphs         = True
    model = torch.compile(
        model,
        mode="max-autotune",
        dynamic=False,
        fullgraph=True
    )

    loss_fn   = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.3,
        patience=20,
        min_lr=1e-7,
        verbose=True
    )
    scaler      = GradScaler()
    best_mIoU   = -1.0
    train_history, val_history = [], []

    for epoch in range(Config.NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{Config.NUM_EPOCHS} ---")
        train_mIoU = train_fn(train_loader, model, optimizer, loss_fn, scaler)
        val_loss   = validate_fn(val_loader, model, loss_fn)
        current_mIoU, current_dice = check_metrics(
            val_loader, model,
            n_classes=6,
            device=Config.DEVICE
        )
        scheduler.step(val_loss)

        train_history.append(train_mIoU.item())
        val_history.append(current_mIoU.item())

        if current_mIoU > best_mIoU:
            best_mIoU = current_mIoU
            print(f"🔹 Nuevo mejor mIoU: {best_mIoU:.4f} | Dice: {current_dice:.4f} → guardando…")
            torch.save({
                "epoch":      epoch,
                "state_dict": model.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "best_mIoU":  best_mIoU,
            }, Config.MODEL_SAVE_PATH)

    save_performance_plot(
        train_history=train_history,
        val_history=val_history,
        save_path="/content/drive/MyDrive/colab/rendimiento_miou.png"
    )

    print("\nEvaluando el mejor modelo guardado…")
    checkpoint = torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    best_mIoU, best_dice = check_metrics(
        val_loader, model,
        n_classes=6,
        device=Config.DEVICE
    )
    print(f"mIoU del modelo guardado: {best_mIoU:.4f} | Dice: {best_dice:.4f}")

if __name__ == "__main__":
    main()
