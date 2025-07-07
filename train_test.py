import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
from PIL import Image
import numpy as np
# Importa la arquitectura del otro archivo
from model3 import CloudDeepLabV3Plus
from utils import imprimir_distribucion_clases_post_augmentation
from config import Config

# =================================================================================
# 2. DATASET PERSONALIZADO (MODIFICADO)
# =================================================================================
class CloudDataset(torch.utils.data.Dataset):
    _IMG_EXTENSIONS = ('.jpg', '.png')

    def __init__(self, image_dir: str, mask_dir: str, transform: A.Compose | None = None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith(self._IMG_EXTENSIONS)
        ]

    def __len__(self) -> int:
        return len(self.images)

    def _mask_path_from_image_name(self, image_filename: str) -> str:
        name_without_ext = image_filename.rsplit('.', 1)[0]
        mask_filename = f"{name_without_ext}_mask.png"
        return os.path.join(self.mask_dir, mask_filename)

    def __getitem__(self, idx: int):
        img_filename = self.images[idx]
        img_path = os.path.join(self.image_dir, img_filename)
        mask_path = self._mask_path_from_image_name(img_filename)
        
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Máscara no encontrada para {img_filename} en {mask_path}")

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask

# =================================================================================
# 3. FUNCIONES DE ENTRENAMIENTO Y VALIDACIÓN (Sin cambios)
# =================================================================================
def train_fn(loader, model, optimizer, loss_fn, scaler, num_classes=6):
    """Procesa una época de entrenamiento con cálculo de IoU y Dice por clase."""
    loop = tqdm(loader, leave=True)
    model.train()

    tp = torch.zeros(num_classes, device=Config.DEVICE)
    fp = torch.zeros(num_classes, device=Config.DEVICE)
    fn = torch.zeros(num_classes, device=Config.DEVICE)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(Config.DEVICE, non_blocking=True)
        targets = targets.to(Config.DEVICE, non_blocking=True).long()

        with autocast(device_type=Config.DEVICE, dtype=torch.float16):
            output = model(data)
            predictions = output[0] if isinstance(output, tuple) else output
            loss = loss_fn(predictions, targets)

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

    epsilon = 1e-6
    iou_per_class = tp / (tp + fp + fn + epsilon)
    dice_per_class = (2 * tp) / (2 * tp + fp + fn + epsilon)

    mean_iou = torch.nanmean(iou_per_class)
    mean_dice = torch.nanmean(dice_per_class)

    print("\nÉpoca de entrenamiento finalizada:")
    print("Dice por clase:", dice_per_class)
    print("IoU por clase:", iou_per_class)
    print("mIoU:", mean_iou)
    print("mDice:", mean_dice)

def check_metrics(loader, model, n_classes=6, device="cuda"):
    model.eval()
    # inicializa contadores en GPU
    tp = torch.zeros(n_classes, device=device, dtype=torch.long)
    fp = torch.zeros(n_classes, device=device, dtype=torch.long)
    fn = torch.zeros(n_classes, device=device, dtype=torch.long)

    with torch.inference_mode():
        for x, y in loader:
            # carga batch en GPU
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).long()

            # inferencia y predicción
            out    = model(x)
            logits = out[0] if isinstance(out, tuple) else out
            preds  = logits.argmax(dim=1)

            # vectoriza métricas con one-hot
            ph = F.one_hot(preds, num_classes=n_classes).view(-1, n_classes)
            yh = F.one_hot(y,     num_classes=n_classes).view(-1, n_classes)

            tp += (ph & yh).sum(dim=0)
            fp += (ph & ~yh).sum(dim=0)
            fn += (~ph & yh).sum(dim=0)

    # Cálculo de IoU y Dice en GPU
    inter      = tp.float()
    sum_pred   = (tp + fp).float()
    sum_truth  = (tp + fn).float()
    union      = sum_pred + sum_truth - inter

    iou_per_class  = inter / (union + 1e-6)
    dice_per_class = (2 * inter) / (sum_pred + sum_truth + 1e-6)

    miou_macro = iou_per_class.mean()
    dice_macro = dice_per_class.mean()

    print("IoU por clase:", iou_per_class)
    print("Dice por clase:", dice_per_class)
    print("Mean IoU (macro):", miou_macro)
    print("Mean Dice (macro):", dice_macro)

    model.train()
    return miou_macro, dice_macro

# =================================================================================
# 4. FUNCIÓN PRINCIPAL DE EJECUCIÓN (Sin cambios)
# =================================================================================
def main():
    torch.backends.cudnn.benchmark = True

    print(f"Using device: {Config.DEVICE}")
    
    train_transform = A.Compose([
        A.Resize(height=Config.IMAGE_HEIGHT, width=Config.IMAGE_WIDTH),
        A.Rotate(limit=35, p=0.7),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Normalize(mean=[0.0,0.0,0.0], std=[1.0,1.0,1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(height=Config.IMAGE_HEIGHT, width=Config.IMAGE_WIDTH),
        A.Normalize(mean=[0.0,0.0,0.0], std=[1.0,1.0,1.0], max_pixel_value=255.0),
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
        shuffle=True
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

    imprimir_distribucion_clases_post_augmentation(train_loader, 6,
        "Distribución de clases en ENTRENAMIENTO (post-aug)")

    model = CloudDeepLabV3Plus(num_classes=6).to(Config.DEVICE)
    print("Compiling the model... (this may take a minute)")
    torch._inductor.config.triton.cudagraphs = True
    model = torch.compile(model)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scaler = GradScaler() 
    best_mIoU = -1.0

    for epoch in range(Config.NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{Config.NUM_EPOCHS} ---")
        print("Calculando métricas de entrenamiento...")
        train_fn(train_loader, model, optimizer, loss_fn, scaler)
        print("Calculando métricas de validación...")
        current_mIoU, current_dice = check_metrics(val_loader, model, n_classes=6, device=Config.DEVICE)
        print("Verificando mejor Mean IoU...")

        if current_mIoU > best_mIoU:
            best_mIoU = current_mIoU
            print(f"🔹 Nuevo mejor mIoU: {best_mIoU:.4f} | Dice: {current_dice:.4f}  →  guardando modelo…")
            checkpoint = {
                "epoch":      epoch,
                "state_dict": model.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "best_mIoU":  best_mIoU,
            }
            torch.save(checkpoint, Config.MODEL_SAVE_PATH)

    print("\nEvaluando el modelo con mejor mIoU guardado…")
    best_model_checkpoint = torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
    model.load_state_dict(best_model_checkpoint['state_dict'])
    best_mIoU, best_dice = check_metrics(val_loader, model, n_classes=6, device=Config.DEVICE)
    print(f"mIoU del modelo guardado: {best_mIoU:.4f} | Dice: {best_dice:.4f}")

if __name__ == "__main__":
    main()