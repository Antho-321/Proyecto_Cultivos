import os
import random

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler
from torch.utils.data import DataLoader

import albumentations as A
from albumentations.core.transforms_interface import DualTransform
from albumentations.pytorch import ToTensorV2

# Importa la arquitectura del otro archivo
from model import CloudDeepLabV3Plus 

# =================================================================================
# 1. CONFIGURACIÓN
# =================================================================================
class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Rutas de datos
    TRAIN_IMG_DIR    = "Balanced/train/images"
    TRAIN_MASK_DIR   = "Balanced/train/masks"
    VAL_IMG_DIR      = "Balanced/val/images"
    VAL_MASK_DIR     = "Balanced/val/masks"
    
    # Hiperparámetros
    LEARNING_RATE    = 1e-4
    BATCH_SIZE       = 8
    NUM_EPOCHS       = 50
    NUM_WORKERS      = 2
    
    # Dimensiones
    IMAGE_HEIGHT     = 256  # tamaño final que espera el modelo
    IMAGE_WIDTH      = 256
    CROP_SIZE        = 96   # NUEVO: tamaño del crop

    # Misc
    PIN_MEMORY       = True
    LOAD_MODEL       = False
    MODEL_SAVE_PATH  = "best_model.pth.tar"

# =================================================================================
# 2. TRANSFORMACIÓN CUSTOM: ClassAwareRandomCrop
#    Intenta hasta max_tries recortes aleatorios de tamaño (h, w) que incluyan
#    al menos un píxel de la clase `class_id`. Si no lo consigue, hace un crop
#    aleatorio normal.
# =================================================================================
class ClassAwareRandomCrop(DualTransform):
    def __init__(self, height, width, class_id=4, max_tries=10, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.height    = height
        self.width     = width
        self.class_id  = class_id
        self.max_tries = max_tries

    def get_params(self):
        return {"top": None, "left": None}

    def apply(self, img, top=0, left=0, **params):
        return img[top : top + self.height, left : left + self.width]

    def apply_to_mask(self, mask, top=0, left=0, **params):
        return mask[top : top + self.height, left : left + self.width]

    def get_transform_init_args_names(self):
        return ("height", "width", "class_id", "max_tries")

    def get_params_dependent_on_targets(self, params):
        mask = params["mask"]
        h, w = mask.shape[:2]

        # si la máscara es más pequeña que el crop, antes se debe redimensionar...
        if h < self.height or w < self.width:
            top, left = 0, 0
        else:
            # intentar hasta max_tries encontrar crop con class_id
            for _ in range(self.max_tries):
                top  = random.randint(0, h - self.height)
                left = random.randint(0, w - self.width)
                patch = mask[top : top + self.height, left : left + self.width]
                if (patch == self.class_id).any():
                    break
            else:
                # fallback: crop aleatorio
                top  = random.randint(0, h - self.height)
                left = random.randint(0, w - self.width)
        return {"top": top, "left": left}

# =================================================================================
# 3. DATASET PERSONALIZADO
# =================================================================================
class CloudDataset(torch.utils.data.Dataset):
    _IMG_EXTENSIONS = ('.jpg', '.png')

    def __init__(self, image_dir: str, mask_dir: str, transform: A.Compose | None = None):
        self.image_dir = image_dir
        self.mask_dir  = mask_dir
        self.transform = transform

        self.images = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith(self._IMG_EXTENSIONS)
        ]

    def __len__(self) -> int:
        return len(self.images)

    def _mask_path_from_image_name(self, image_filename: str) -> str:
        name_without_ext = image_filename.rsplit('.', 1)[0]
        mask_filename   = f"{name_without_ext}_mask.png"
        return os.path.join(self.mask_dir, mask_filename)

    def __getitem__(self, idx: int):
        img_fname = self.images[idx]
        img_path  = os.path.join(self.image_dir, img_fname)
        mask_path = self._mask_path_from_image_name(img_fname)

        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Máscara no encontrada para {img_fname}")

        # Carga
        image = np.array(Image.open(img_path).convert("RGB"))
        mask  = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)

        # Transform
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask  = augmented["mask"]

        return image, mask

# =================================================================================
# 4. FUNCIONES DE ENTRENAMIENTO Y VALIDACIÓN (sin cambios)
# =================================================================================
def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader, leave=True)
    model.train()
    for data, targets in loop:
        data    = data.to(Config.DEVICE, non_blocking=True)
        targets = targets.to(Config.DEVICE, non_blocking=True).long()

        with torch.cuda.amp.autocast():
            preds = model(data)
            loss  = loss_fn(preds, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())

def check_metrics(loader, model, n_classes=6, device="cuda"):
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
    miou_macro     = iou_per_class.mean()
    dice_macro     = dice_per_class.mean()

    print("IoU por clase :", iou_per_class.cpu().numpy())
    print("Dice por clase:", dice_per_class.cpu().numpy())
    print(f"mIoU macro = {miou_macro:.4f} | Dice macro = {dice_macro:.4f}")

    model.train()
    return miou_macro, dice_macro

# =================================================================================
# 5. MAIN
# =================================================================================
def main():
    print(f"Using device: {Config.DEVICE}")

    train_transform = A.Compose([
        # 1) Crop de 96×96 garantizando presencia de la clase 4
        ClassAwareRandomCrop(
            height=Config.CROP_SIZE,
            width= Config.CROP_SIZE,
            class_id=4,
            max_tries=10,
            p=1.0
        ),
        # 2) Redimensionar de vuelta a (256×256)
        A.Resize(
            height=Config.IMAGE_HEIGHT,
            width= Config.IMAGE_WIDTH
        ),
        # 3) Resto de aumentos
        A.Rotate(limit=35, p=0.7),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std =[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        # Para validación se mantiene el resize directo
        A.Resize(height=Config.IMAGE_HEIGHT, width=Config.IMAGE_WIDTH),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std =[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])

    # --- DATASETS Y DATALOADERS ---
    train_dataset = CloudDataset(Config.TRAIN_IMG_DIR, Config.TRAIN_MASK_DIR, transform=train_transform)
    train_loader  = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    val_dataset = CloudDataset(Config.VAL_IMG_DIR, Config.VAL_MASK_DIR, transform=val_transform)
    val_loader  = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    # --- MODELO, LOSS, OPTIMIZADOR, SCALER ---
    model     = CloudDeepLabV3Plus(num_classes=6).to(Config.DEVICE)
    loss_fn   = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scaler    = GradScaler()

    best_mIoU = -1.0

    # --- TRAINING LOOP ---
    for epoch in range(Config.NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{Config.NUM_EPOCHS} ---")
        train_fn(train_loader, model, optimizer, loss_fn, scaler)
        current_mIoU, current_dice = check_metrics(val_loader, model, n_classes=6, device=Config.DEVICE)

        if current_mIoU > best_mIoU:
            best_mIoU = current_mIoU
            print(f"🔹 Nuevo mejor mIoU: {best_mIoU:.4f} | Dice: {current_dice:.4f} → guardando modelo…")
            checkpoint = {
                "epoch":      epoch,
                "state_dict": model.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "best_mIoU":  best_mIoU,
            }
            torch.save(checkpoint, Config.MODEL_SAVE_PATH)

    print("\nEvaluando el mejor modelo guardado…")
    best_mIoU, best_dice = check_metrics(val_loader, model, n_classes=6, device=Config.DEVICE)
    print(f"mIoU del modelo guardado: {best_mIoU:.4f} | Dice: {best_dice:.4f}")

if __name__ == "__main__":
    main()
