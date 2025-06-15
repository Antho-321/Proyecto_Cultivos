# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
from PIL import Image
import numpy as np

# Importa la arquitectura del otro archivo
from model import CloudDeepLabV3Plus
from distribucion_por_clase import imprimir_distribucion_clases_post_augmentation 

# ---------------------------------------------------------------------------------
# 1. CONFIGURACIÓN  (añadimos dos hiperparámetros nuevos)
# ---------------------------------------------------------------------------------
class Config:
    DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"

    TRAIN_IMG_DIR     = "Balanced/train/images"
    TRAIN_MASK_DIR    = "Balanced/train/masks"
    VAL_IMG_DIR       = "Balanced/val/images"
    VAL_MASK_DIR      = "Balanced/val/masks"

    LEARNING_RATE     = 1e-4
    BATCH_SIZE        = 8
    NUM_EPOCHS        = 50
    NUM_WORKERS       = 2

    IMAGE_HEIGHT      = 256
    IMAGE_WIDTH       = 256

    # —— NUEVO ——
    CLASS4_WEIGHT     = 6      # cuántas veces “vale” una imagen con clase 4
    CROP_P_ALWAYS     = 1.0    # fuerza el CropAroundClass4 cuando haya píxeles de clase 4

    PIN_MEMORY        = True
    LOAD_MODEL        = False
    MODEL_SAVE_PATH   = "best_model.pth.tar"

# ---------------------------------------------------------------------------------
# 2. DATASET PERSONALIZADO con flag ‘contains_class4’
# ---------------------------------------------------------------------------------
class CloudDataset(torch.utils.data.Dataset):
    _IMG_EXTENSIONS = ('.jpg', '.png')

    def __init__(self, image_dir: str, mask_dir: str, transform: A.Compose | None = None):
        self.image_dir  = image_dir
        self.mask_dir   = mask_dir
        self.transform  = transform
        self.images     = [f for f in os.listdir(image_dir)
                           if f.lower().endswith(self._IMG_EXTENSIONS)]

        # Pre-escaneamos las máscaras UNA sola vez para saber si incluyen clase 4
        self.contains_class4 = []
        for img_file in self.images:
            mask_path = self._mask_path_from_image_name(img_file)
            mask = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
            self.contains_class4.append((mask == 4).any())

    def __len__(self):  return len(self.images)

    def _mask_path_from_image_name(self, image_filename: str) -> str:
        name = image_filename.rsplit('.', 1)[0]
        return os.path.join(self.mask_dir, f"{name}_mask.png")

    def __getitem__(self, idx: int):
        img_path  = os.path.join(self.image_dir, self.images[idx])
        mask_path = self._mask_path_from_image_name(self.images[idx])

        image = np.array(Image.open(img_path).convert("RGB"))
        mask  = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)

        if self.transform:
            aug = self.transform(image=image, mask=mask)
            image, mask = aug["image"], aug["mask"]

        return image, mask

class SmartCrop(A.DualTransform):
    def __init__(self, crop_size=(64, 64), p=0.7, focus_cls=4, max_tries=30):
        super().__init__(always_apply=False, p=p)
        self.ch, self.cw  = crop_size
        self.focus_cls    = focus_cls
        self.max_tries    = max_tries

    def get_params_dependent_on_targets(self, params):
        mask = params["mask"]
        h, w = mask.shape[:2]

        # Todas las coords de foreground
        ys_fg, xs_fg = np.where(mask != 0)
        if len(ys_fg) == 0:
            return {"y1": 0, "x1": 0, "skip": True}  # máscara vacía

        # Si hay píxeles de la clase de interés, priorízalos
        ys_focus, xs_focus = np.where(mask == self.focus_cls)
        candidate_coords   = list(zip(ys_focus, xs_focus)) if len(ys_focus) else list(zip(ys_fg, xs_fg))

        for _ in range(self.max_tries):
            cy, cx = candidate_coords[np.random.randint(len(candidate_coords))]
            y1     = int(np.clip(cy - self.ch//2, 0, h - self.ch))
            x1     = int(np.clip(cx - self.cw//2, 0, w - self.cw))
            crop   = mask[y1:y1+self.ch, x1:x1+self.cw]
            if (crop == 0).any():   # todavía hay fondo → intenta de nuevo
                continue
            return {"y1": y1, "x1": x1, "skip": False}
        # Si no encontró ninguno, mantén la imagen original
        return {"y1": 0, "x1": 0, "skip": True}

    def apply(self, img, skip=False, y1=0, x1=0, **params):
        return img if skip else img[y1:y1+self.ch, x1:x1+self.cw]

    def apply_to_mask(self, mask, **kw):
        return self.apply(mask, **kw)

# =================================================================================
# 3. FUNCIONES DE ENTRENAMIENTO Y VALIDACIÓN
# =================================================================================
def train_fn(loader, model, optimizer, loss_fn, scaler):
    """Procesa una época de entrenamiento."""
    loop = tqdm(loader, leave=True)
    model.train()

    for batch_idx, (data, targets) in enumerate(loop):
        data     = data.to(Config.DEVICE, non_blocking=True)
        targets  = targets.to(Config.DEVICE, non_blocking=True).long()  # <- importante

        # Forward
        with autocast(device_type='cuda', dtype=torch.float16):
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # Backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Actualizar la barra de progreso
        loop.set_postfix(loss=loss.item())

def check_metrics(loader, model, n_classes=6, device="cuda"):
    """
    Devuelve mIoU macro y Dice macro.
    Imprime IoU y Dice por clase.
    """
    eps = 1e-8                          # para evitar divisiones por 0
    intersection_sum = torch.zeros(n_classes, dtype=torch.float64, device=device)
    union_sum        = torch.zeros_like(intersection_sum)
    dice_num_sum     = torch.zeros_like(intersection_sum)   # 2*intersección
    dice_den_sum     = torch.zeros_like(intersection_sum)   # pred + gt

    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).long()      # (N, H, W)

            logits = model(x)                               # (N, 6, H, W)
            preds  = torch.argmax(logits, dim=1)            # (N, H, W)

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

    # IoU y Dice por clase
    iou_per_class   = (intersection_sum + eps) / (union_sum + eps)
    dice_per_class  = (dice_num_sum   + eps) / (dice_den_sum + eps)

    # Promedios macro
    miou_macro  = iou_per_class.mean()
    dice_macro  = dice_per_class.mean()

    print("IoU por clase :", iou_per_class.cpu().numpy())
    print("Dice por clase:", dice_per_class.cpu().numpy())
    print(f"mIoU macro = {miou_macro:.4f} | Dice macro = {dice_macro:.4f}")

    model.train()
    return miou_macro, dice_macro

# =================================================================================
# 4. FUNCIÓN PRINCIPAL DE EJECUCIÓN
# =================================================================================
def main():
    print(f"Using device: {Config.DEVICE}")
    
    # --- Transformaciones y Aumento de Datos ---
    train_transform = A.Compose([
        SmartCrop(crop_size=(64, 64), p=0.7),      # ← único crop
        A.Rotate(limit=35, p=0.7, border_mode=0, value=0, mask_value=0),  # sigue añadiendo algo de 0 pero muucho menos
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Normalize(mean=[0,0,0], std=[1,1,1]),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Resize(height=Config.IMAGE_HEIGHT, width=Config.IMAGE_WIDTH),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255),
        ToTensorV2()
    ])

    # --- Creación de Datasets y DataLoaders ---
    train_dataset = CloudDataset(Config.TRAIN_IMG_DIR, Config.TRAIN_MASK_DIR, transform=train_transform)

    # --- Ponderaciones: alto peso si la imagen tiene clase 4, 1 en caso contrario ---
    weights = [
        Config.CLASS4_WEIGHT if has_c4 else 1
        for has_c4 in train_dataset.contains_class4
    ]

    # Mantenemos ‘replacement=True’ para que una misma imagen pueda salir varias veces en una época
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights) * Config.CLASS4_WEIGHT,  # p. ej. 210 × 6 = 1260
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        sampler=sampler,          # << NO usar shuffle cuando hay sampler
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )

    val_dataset   = CloudDataset(Config.VAL_IMG_DIR,   Config.VAL_MASK_DIR,   transform=val_transform)

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )

    imprimir_distribucion_clases_post_augmentation(
        train_loader, 
        n_classes=6, 
        title="Distribución de clases en el set de ENTRENAMIENTO (post-augmentation)"
    )
    
    # --- Instanciación del Modelo, Loss y Optimizador ---
    model = CloudDeepLabV3Plus(num_classes=6).to(Config.DEVICE)
    
    loss_fn = nn.CrossEntropyLoss()
    
    # AdamW es una buena elección de optimizador por defecto.
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # El scaler es para el entrenamiento de precisión mixta (acelera el entrenamiento en GPUs compatibles)
    scaler = GradScaler()

    best_mIoU = -1.0

    # --- Bucle Principal de Entrenamiento ---
    for epoch in range(Config.NUM_EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{Config.NUM_EPOCHS} ---")

        # 1) Entrenamiento de una época
        train_fn(
            train_loader,
            model,
            optimizer,
            loss_fn,
            scaler
        )

        # 2) Evaluación en el conjunto de validación
        current_mIoU, current_dice = check_metrics(
            val_loader,
            model,
            n_classes=6,
            device=Config.DEVICE
        )

        # 3) Guardar checkpoint si hubo mejora en mIoU
        if current_mIoU > best_mIoU:
            best_mIoU = current_mIoU
            print(f"🔹 Nuevo mejor mIoU: {best_mIoU:.4f} | Dice: {current_dice:.4f}  →  guardando modelo…")

            checkpoint = {
                "epoch":     epoch,
                "state_dict": model.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "best_mIoU":  best_mIoU,
            }
            torch.save(checkpoint, Config.MODEL_SAVE_PATH)

    print("\nEvaluando el modelo con mejor mIoU guardado…")
    best_mIoU, best_dice = check_metrics(
        val_loader,
        model,
        n_classes=6,
        device=Config.DEVICE
    )
    print(f"mIoU del modelo guardado: {best_mIoU:.4f} | Dice: {best_dice:.4f}")

if __name__ == "__main__":
    main()