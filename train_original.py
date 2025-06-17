# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
from PIL import Image
import numpy as np

# Importa la arquitectura del otro archivo
from model import CloudDeepLabV3Plus

# =================================================================================
# 1. CONFIGURACIÓN
# =================================================================================
class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    TRAIN_IMG_DIR = "Balanced/train/images"
    TRAIN_MASK_DIR = "Balanced/train/masks"
    VAL_IMG_DIR = "Balanced/val/images"
    VAL_MASK_DIR = "Balanced/val/masks"
    
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 8
    NUM_EPOCHS = 50
    NUM_WORKERS = 2
    
    IMAGE_HEIGHT = 256
    IMAGE_WIDTH = 256
    
    PIN_MEMORY = True
    LOAD_MODEL = False
    MODEL_SAVE_PATH = "best_model.pth.tar"

# =================================================================================
# FUNCIÓN DE RECORTE (AÑADIDA)
# =================================================================================
def crop_around_classes(
    image: np.ndarray,
    mask: np.ndarray,
    classes_to_find: list[int] = [1, 2, 3, 4, 5],
    margin: int = 10  # <--- AÑADIDO: Un pequeño margen puede ser útil
) -> tuple[np.ndarray, np.ndarray]:
    """
    Recorta un rectángulo alrededor de todos los píxeles que pertenecen a las
    clases especificadas en la máscara.
    """
    # is_class_present será 2D (H,W) ya que la máscara de entrada es (H,W,1)
    is_class_present = np.isin(mask.squeeze(), classes_to_find)

    ys, xs = np.where(is_class_present)

    if ys.size == 0:
        return image, mask # Devuelve la máscara original (H,W,1)

    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()

    y0 = max(0, y_min - margin)
    y1 = min(mask.shape[0], y_max + margin + 1)
    x0 = max(0, x_min - margin)
    x1 = min(mask.shape[1], x_max + margin + 1)

    cropped_image = image[y0:y1, x0:x1]
    # Se recorta la máscara (H,W,1) y mantiene sus 3 dimensiones
    cropped_mask = mask[y0:y1, x0:x1, :]

    return cropped_image, cropped_mask

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

        # --- MODIFICACIÓN CLAVE: Aplicar recorte ANTES de las transformaciones ---
        # 1. Añadir una dimensión de canal a la máscara para que sea (H, W, 1)
        mask_3d = np.expand_dims(mask, axis=-1)
        
        # 2. Aplicar la función de recorte
        image_cropped, mask_cropped_3d = crop_around_classes(image, mask_3d)

        # 3. Quitar la dimensión del canal de la máscara para Albumentations
        mask_cropped = mask_cropped_3d.squeeze()
        # ------------------------------------------------------------------------

        if self.transform:
            # Pasa los arrays RECORTADOS a las transformaciones
            augmented = self.transform(image=image_cropped, mask=mask_cropped)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask

# =================================================================================
# 3. FUNCIONES DE ENTRENAMIENTO Y VALIDACIÓN (Sin cambios)
# ... (El resto de tu código: train_fn, check_metrics)
# =================================================================================
def train_fn(loader, model, optimizer, loss_fn, scaler):
    """Procesa una época de entrenamiento."""
    loop = tqdm(loader, leave=True)
    model.train()

    for batch_idx, (data, targets) in enumerate(loop):
        data     = data.to(Config.DEVICE, non_blocking=True)
        targets  = targets.to(Config.DEVICE, non_blocking=True).long()

        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())

def check_metrics(loader, model, n_classes=6, device="cuda"):
    """Devuelve mIoU macro y Dice macro."""
    eps = 1e-8
    intersection_sum = torch.zeros(n_classes, dtype=torch.float64, device=device)
    union_sum        = torch.zeros_like(intersection_sum)
    dice_num_sum     = torch.zeros_like(intersection_sum)
    dice_den_sum     = torch.zeros_like(intersection_sum)

    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).long()

            logits = model(x)
            preds  = torch.argmax(logits, dim=1)

            for cls in range(n_classes):
                pred_c   = (preds == cls)
                true_c   = (y == cls)
                inter    = (pred_c & true_c).sum().double()
                pred_sum = pred_c.sum().double()
                true_sum = true_c.sum().double()
                union    = pred_sum + true_sum - inter

                intersection_sum[cls] += inter
                union_sum[cls]        += union
                dice_num_sum[cls]     += 2 * inter
                dice_den_sum[cls]     += pred_sum + true_sum

    iou_per_class  = (intersection_sum + eps) / (union_sum + eps)
    dice_per_class = (dice_num_sum   + eps) / (dice_den_sum + eps)

    miou_macro = iou_per_class.mean()
    dice_macro = dice_per_class.mean()

    print("IoU por clase :", iou_per_class.cpu().numpy())
    print("Dice por clase:", dice_per_class.cpu().numpy())
    print(f"mIoU macro = {miou_macro:.4f} | Dice macro = {dice_macro:.4f}")

    model.train()
    return miou_macro, dice_macro


# =================================================================================
# 4. FUNCIÓN PRINCIPAL DE EJECUCIÓN (Sin cambios)
# =================================================================================
def main():
    print(f"Using device: {Config.DEVICE}")
    
    train_transform = A.Compose([
        A.Resize(height=Config.IMAGE_HEIGHT, width=Config.IMAGE_WIDTH), # <-- MUY IMPORTANTE
        A.Rotate(limit=35, p=0.7),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(height=Config.IMAGE_HEIGHT, width=Config.IMAGE_WIDTH), # <-- MUY IMPORTANTE
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
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
    
    model = CloudDeepLabV3Plus(num_classes=6).to(Config.DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scaler = GradScaler() 
    best_mIoU = -1.0

    for epoch in range(Config.NUM_EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{Config.NUM_EPOCHS} ---")
        train_fn(train_loader, model, optimizer, loss_fn, scaler)
        current_mIoU, current_dice = check_metrics(val_loader, model, n_classes=6, device=Config.DEVICE)

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
    best_mIoU, best_dice = check_metrics(val_loader, model, n_classes=6, device=Config.DEVICE)
    print(f"mIoU del modelo guardado: {best_mIoU:.4f} | Dice: {best_dice:.4f}")

if __name__ == "__main__":
    main()