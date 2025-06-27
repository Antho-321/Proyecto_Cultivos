# train.py

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
# Importa la arquitectura del otro archivo
from model import CloudDeepLabV3Plus
from utils import imprimir_distribucion_clases_post_augmentation, crop_around_classes, save_performance_plot
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
def train_fn(loader, model, optimizer, loss_fn, scaler, accumulation_steps=4):
    loop = tqdm(loader, leave=True)
    model.train()

    optimizer.zero_grad()
    
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(Config.DEVICE, non_blocking=True)
        targets = targets.to(Config.DEVICE, non_blocking=True).long()

        # Try removing autocast to see if it resolves the issue
        # with autocast(device_type=Config.DEVICE, dtype=torch.float16): 
        output = model(data)
        predictions = output[0] if isinstance(output, tuple) else output
        loss = loss_fn(predictions, targets)

        loss.backward()
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

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

            output = model(x)
            logits = output[0] if isinstance(output, tuple) else output # <-- ¡ESTA ES LA CORRECCIÓN CLAVE!
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
torch.backends.cudnn.benchmark = True

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

    imprimir_distribucion_clases_post_augmentation(train_loader, 6,
        "Distribución de clases en ENTRENAMIENTO (post-aug)")

    model = CloudDeepLabV3Plus(num_classes=6).to(Config.DEVICE)
    print("Compiling the model... (this may take a minute)")
    model = torch.compile(model, mode="max-autotune")
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scaler = GradScaler() 
    best_mIoU = -1.0

    # --- 2. INICIALIZAR LISTAS PARA EL HISTORIAL ---
    train_miou_history = []
    val_miou_history = []

    for epoch in range(Config.NUM_EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{Config.NUM_EPOCHS} ---")
        train_fn(train_loader, model, optimizer, loss_fn, scaler)
        
        # --- 3. CALCULAR MÉTRICAS PARA ENTRENAMIENTO Y VALIDACIÓN ---
        print("Calculando métricas de entrenamiento...")
        train_mIoU, _ = check_metrics(train_loader, model, n_classes=6, device=Config.DEVICE)
        
        print("Calculando métricas de validación...")
        current_mIoU, current_dice = check_metrics(val_loader, model, n_classes=6, device=Config.DEVICE)

        # --- 4. GUARDAR LAS MÉTRICAS EN EL HISTORIAL ---
        train_miou_history.append(train_mIoU.item()) # .item() para obtener el valor escalar
        val_miou_history.append(current_mIoU.item())

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

    # --- 5. LLAMAR A LA FUNCIÓN DE GRAFICADO AL FINALIZAR ---
    save_performance_plot(
        train_history=train_miou_history,
        val_history=val_miou_history,
        save_path="/content/drive/MyDrive/colab/rendimiento_miou.png"
    )

    print("\nEvaluando el modelo con mejor mIoU guardado…")

    # --- Cargar el checkpoint del mejor modelo ---
    # Añadir map_location para asegurar compatibilidad entre CPU/GPU
    best_model_checkpoint = torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
    model.load_state_dict(best_model_checkpoint['state_dict'])

    # Ahora que el mejor modelo está cargado, se ejecuta la evaluación
    best_mIoU, best_dice = check_metrics(val_loader, model, n_classes=6, device=Config.DEVICE)
    print(f"mIoU del modelo guardado: {best_mIoU:.4f} | Dice: {best_dice:.4f}")
if __name__ == "__main__":
    main()