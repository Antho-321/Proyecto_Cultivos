import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
from PIL import Image
import numpy as np

# Arquitectura y utilidades propias
from model import CloudDeepLabV3Plus
from utils import (
    imprimir_distribucion_clases_post_augmentation,
    crop_around_classes,
    save_performance_plot,
)
from config import Config

# --- Aceleradores globales ------------------------------------------------------
torch.backends.cuda.matmul.allow_tf32 = True  # kernels TF32 en Ampere+
torch.backends.cudnn.benchmark = True         # autotune convolutions
torch.set_float32_matmul_precision("high")    # PyTorch 2.3+
from torch.optim.lr_scheduler import ReduceLROnPlateau

# =================================================================================
# 1. DATASET PERSONALIZADO
# =================================================================================
class CloudDataset(torch.utils.data.Dataset):
    _IMG_EXTENSIONS = ('.jpg', '.png')

    def __init__(self, image_dir: str, mask_dir: str, transform: A.Compose | None = None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = [f for f in os.listdir(image_dir) if f.lower().endswith(self._IMG_EXTENSIONS)]

    def __len__(self) -> int:
        return len(self.images)

    def _mask_path_from_image_name(self, image_filename: str) -> str:
        name_without_ext = image_filename.rsplit('.', 1)[0]
        return os.path.join(self.mask_dir, f"{name_without_ext}_mask.png")

    def __getitem__(self, idx: int):
        img_filename = self.images[idx]
        img_path  = os.path.join(self.image_dir, img_filename)
        mask_path = self._mask_path_from_image_name(img_filename)

        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Máscara no encontrada para {img_filename} en {mask_path}")

        image = np.array(Image.open(img_path).convert("RGB"))
        mask  = np.array(Image.open(mask_path).convert("L"))

        # ── Recorte centrado en la(s) clase(s) PRESENTE(S) ────────────────────
        mask_3d = np.expand_dims(mask, axis=-1)
        image_cropped, mask_cropped_3d = crop_around_classes(image, mask_3d)
        mask_cropped = mask_cropped_3d.squeeze()
        # ---------------------------------------------------------------------

        if self.transform:
            augmented = self.transform(image=image_cropped, mask=mask_cropped)
            image = augmented["image"]
            mask  = augmented["mask"]

        return image, mask

# =================================================================================
# 2. ENTRENAMIENTO Y VALIDACIÓN
# =================================================================================

def train_fn(loader, model, optimizer, loss_fn, scaler, num_classes: int = 6):
    """Procesa una época de entrenamiento con cálculo de IoU y Dice por clase."""
    loop = tqdm(loader, leave=True)
    model.train()

    tp = torch.zeros(num_classes, device=Config.DEVICE)
    fp = torch.zeros(num_classes, device=Config.DEVICE)
    fn = torch.zeros(num_classes, device=Config.DEVICE)

    for data, targets in loop:
        data     = data.to(Config.DEVICE, non_blocking=True)
        targets  = targets.to(Config.DEVICE, non_blocking=True).long()

        with autocast(device_type=Config.DEVICE, dtype=torch.float16):
            output      = model(data)
            predictions = output[0] if isinstance(output, tuple) else output
            loss        = loss_fn(predictions, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Métricas
        _, predicted_classes = torch.max(predictions, dim=1)
        for c in range(num_classes):
            tp[c] += ((predicted_classes == c) & (targets == c)).sum()
            fp[c] += ((predicted_classes == c) & (targets != c)).sum()
            fn[c] += ((predicted_classes != c) & (targets == c)).sum()

        loop.set_postfix(loss=loss.item())

    eps = 1e-6
    iou_per_class  = tp / (tp + fp + fn + eps)
    dice_per_class = (2 * tp) / (2 * tp + fp + fn + eps)
    mean_iou = torch.nanmean(iou_per_class)

    print("\nÉpoca finalizada")
    print("  - Dice por clase:", dice_per_class.cpu().numpy())
    print("  - IoU  por clase:", iou_per_class.cpu().numpy())
    print(f"  - mIoU: {mean_iou:.4f}")

    return mean_iou


def check_metrics(loader, model, n_classes: int = 6, device: str = "cuda"):
    eps = 1e-8
    conf_mat = torch.zeros((n_classes, n_classes), dtype=torch.float64, device=device)

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).long()

            logits = model(x)
            logits = logits[0] if isinstance(logits, tuple) else logits
            preds  = torch.argmax(logits, dim=1)

            combined = (preds * n_classes + y).view(-1).float()
            conf = torch.histc(combined, bins=n_classes*n_classes, min=0, max=n_classes*n_classes-1)
            conf_mat += conf.view(n_classes, n_classes)

    inter = torch.diag(conf_mat)
    pred  = conf_mat.sum(dim=1)
    true  = conf_mat.sum(dim=0)
    union = pred + true - inter

    iou  = (inter + eps) / (union + eps)
    dice = (2*inter + eps) / (pred + true + eps)

    print("IoU  por clase:", iou.cpu().numpy())
    print("Dice por clase:", dice.cpu().numpy())
    print(f"mIoU macro = {iou.mean():.4f} | Dice macro = {dice.mean():.4f}")

    model.train()
    return iou.mean(), dice.mean()

# =================================================================================
# 3. FUNCIÓN PRINCIPAL
# =================================================================================

def main():
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(0.95)

    print(f"Using device: {Config.DEVICE}")

    # ── Transformaciones ---------------------------------------------------
    train_transform = A.Compose([
        A.Resize(Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH),
        A.Rotate(limit=35, p=0.7),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Normalize(mean=[0,0,0], std=[1,1,1], max_pixel_value=255.0),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH),
        A.Normalize(mean=[0,0,0], std=[1,1,1], max_pixel_value=255.0),
        ToTensorV2(),
    ])

    # ── Datasets y loaders --------------------------------------------------
    train_loader = DataLoader(
        CloudDataset(Config.TRAIN_IMG_DIR, Config.TRAIN_MASK_DIR, train_transform),
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        shuffle=True,
    )

    val_loader = DataLoader(
        CloudDataset(Config.VAL_IMG_DIR, Config.VAL_MASK_DIR, val_transform),
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        shuffle=False,
    )

    imprimir_distribucion_clases_post_augmentation(train_loader, 6, "Distribución de clases en ENTRENAMIENTO (post-aug)")

    # ── Modelo -------------------------------------------------------------
    model = CloudDeepLabV3Plus(num_classes=6).to(Config.DEVICE)

    # ► Congelar las dos primeras etapas del backbone (stride 2 y 4) ◄
    for idx, block in enumerate(model.backbone):
        if idx < 2:
            block.requires_grad_(False)

    # ── Compilación (Inductor) --------------------------------------------
    torch._inductor.config.triton.unique_kernel_names = True
    torch._inductor.config.epilogue_fusion = "max"
    model = torch.compile(model, mode="max-autotune", dynamic=False, fullgraph=True)

    # Optimizer: solo parámetros con grad = True
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )

    # Si quieres LR diferente para el backbone no congelado, descomenta:
    # backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    # decoder_params  = [p for n, p in model.named_parameters() if "backbone" not in n]
    # optimizer = optim.AdamW([
    #     {"params": decoder_params},
    #     {"params": backbone_params, "lr": Config.LEARNING_RATE * 0.1},
    # ], lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)

    loss_fn  = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=20, min_lr=1e-7, verbose=True)
    scaler = GradScaler()

    best_mIoU = -1.0
    train_history, val_history = [], []

    for epoch in range(Config.NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{Config.NUM_EPOCHS} ---")

        train_miou = train_fn(train_loader, model, optimizer, loss_fn, scaler)
        val_loss   = validate_fn(val_loader, model, loss_fn)
        val_miou, val_dice = check_metrics(val_loader, model, n_classes=6, device=Config.DEVICE)

        scheduler.step(val_loss)
        train_history.append(train_miou.item())
        val_history.append(val_miou.item())

        if val_miou > best_mIoU:
            best_mIoU = val_miou
            print(f"🔹 Nuevo mejor mIoU: {best_mIoU:.4f} | Dice: {val_dice:.4f}  →  guardando modelo…")
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_mIoU": best_mIoU,
            }, Config.MODEL_SAVE_PATH)

        # ── Ejemplo de descongelado progresivo (opcional) ------------------
        # if epoch == 10:  # p.ej. después de 10 épocas
        #     for idx, block in enumerate(model.backbone):
        #         if idx == 1:  # segundo stage
        #             block.requires_grad_(True)
        #     # Re‑crear el optimizador para incluir nuevos params con grad
        #     optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
        #                             lr=Config.LEARNING_RATE * 0.1,
        #                             weight_decay=Config.WEIGHT_DECAY)

    # ── Grafico de rendimiento --------------------------------------------
    save_performance_plot(train_history, val_history, "/content/drive/MyDrive/colab/rendimiento_miou.png")

    # ── Evaluación del mejor checkpoint -----------------------------------
    ckpt = torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
    model.load_state_dict(ckpt['state_dict'])
    best_mIoU, best_dice = check_metrics(val_loader, model, n_classes=6, device=Config.DEVICE)
    print(f"mIoU del modelo guardado: {best_mIoU:.4f} | Dice: {best_dice:.4f}")


def validate_fn(loader, model, loss_fn, device=Config.DEVICE):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True).long()
            logits = model(x)
            logits = logits[0] if isinstance(logits, tuple) else logits
            total_loss += loss_fn(logits, y).item() * x.size(0)

    model.train()
    return total_loss / len(loader.dataset)


if __name__ == "__main__":
    main()
