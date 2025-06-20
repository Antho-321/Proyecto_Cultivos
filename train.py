# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
import os
from PIL import Image
import numpy as np

# Importa la arquitectura del otro archivo
from model import CloudDeepLabV3Plus
from utils import imprimir_distribucion_clases_post_augmentation, save_performance_plot, crop_around_classes
from config import Config
# Se elimina la importación de cv2 ya que solo se usaba en las transformaciones eliminadas

# =================================================================================
# 2. DATASET PERSONALIZADO (MODIFICADO SIN TRANSFORMACIONES)
# =================================================================================
class CloudDataset(torch.utils.data.Dataset):
    """
    Dataset que carga imágenes y máscaras, realiza un recorte y las convierte
    directamente a tensores de PyTorch sin usar librerías de aumentación.
    """
    _IMG_EXTENSIONS = ('.jpg', '.png')

    def __init__(self, image_dir: str, mask_dir: str):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        
        # Simplemente listamos todas las imágenes disponibles
        self.image_files = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith(self._IMG_EXTENSIONS)
        ]
        
        print(f"Dataset inicializado. Número de muestras: {len(self.image_files)}")

    def __len__(self) -> int:
        return len(self.image_files)

    def _mask_path_from_image_name(self, image_filename: str) -> str:
        name_without_ext = image_filename.rsplit('.', 1)[0]
        mask_filename = f"{name_without_ext}_mask.png"
        return os.path.join(self.mask_dir, mask_filename)

    def __getitem__(self, idx: int):
        img_filename = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_filename)
        mask_path = self._mask_path_from_image_name(img_filename)

        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Máscara no encontrada para {img_filename} en {mask_path}")

        # --- LOAD THE IMAGES USING PIL ---
        # Keep them as PIL Images for now to use the resize method easily
        image_pil = Image.open(img_path).convert("RGB")
        mask_pil = Image.open(mask_path).convert("L")
        
        # Convert to NumPy for your cropping logic
        image_np = np.array(image_pil)
        mask_np = np.array(mask_pil)

        # Lógica de recorte (Your existing logic)
        mask_3d = np.expand_dims(mask_np, axis=-1)
        image_cropped_np, mask_cropped_3d = crop_around_classes(image_np, mask_3d)
        mask_cropped_np = mask_cropped_3d.squeeze()

        # --- FIX: RESIZE TO A UNIFORM SIZE ---
        # 1. Convert the cropped NumPy arrays back to PIL Images
        image_cropped_pil = Image.fromarray(image_cropped_np)
        mask_cropped_pil = Image.fromarray(mask_cropped_np)

        # 2. Define your target size
        TARGET_SIZE = (256, 256) # Or (512, 512), etc.

        # 3. Resize both the image and the mask
        # Use NEAREST resampling for the mask to avoid creating new class values (e.g., 0.5)
        image_resized = image_cropped_pil.resize(TARGET_SIZE, Image.Resampling.BILINEAR)
        mask_resized = mask_cropped_pil.resize(TARGET_SIZE, Image.Resampling.NEAREST)

        # --- Convert to Tensores de PyTorch manually ---
        # Now convert the FINAL resized images to NumPy arrays and then to Tensors
        image_final_np = np.array(image_resized)
        mask_final_np = np.array(mask_resized)

        # 1. Imagen: de NumPy (H, W, C) a Tensor (C, H, W) y normalizar a [0, 1]
        image_tensor = torch.from_numpy(image_final_np).float().permute(2, 0, 1) / 255.0

        # 2. Máscara: de NumPy (H, W) a Tensor (H, W) de tipo Long
        mask_tensor = torch.from_numpy(mask_final_np).long()

        return image_tensor, mask_tensor

# =================================================================================
# 3. FUNCIONES DE ENTRENAMIENTO Y VALIDACIÓN (Sin cambios)
# =================================================================================
def train_fn(loader, model, optimizer, loss_fn, scaler):
    """Procesa una época de entrenamiento."""
    loop = tqdm(loader, leave=True)
    model.train()

    for batch_idx, (data, targets) in enumerate(loop):
        data     = data.to(Config.DEVICE, non_blocking=True)
        targets  = targets.to(Config.DEVICE, non_blocking=True).long()

        with torch.cuda.amp.autocast():
            output = model(data)
            predictions = output[0] if isinstance(output, tuple) else output
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

            output = model(x)
            logits = output[0] if isinstance(output, tuple) else output
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
# 4. FUNCIÓN PRINCIPAL DE EJECUCIÓN (MODIFICADA)
# =================================================================================
def main():
    print(f"Using device: {Config.DEVICE}")
    
    # --- 1. CARGAR DATASET ORIGINAL ---
    original_train_dataset = CloudDataset(
        image_dir=Config.TRAIN_IMG_DIR,
        mask_dir=Config.TRAIN_MASK_DIR
    )

    # --- 2. CARGAR DATASET SINTÉTICO (GENERADO POR MGVAE) ---
    # Asumimos que las imágenes y máscaras sintéticas ya fueron generadas y guardadas
    synthetic_img_dir = "data/synthetic_images"
    synthetic_mask_dir = "data/synthetic_masks"

    # Verificamos si existen datos sintéticos para añadirlos
    if os.path.exists(synthetic_img_dir) and os.listdir(synthetic_img_dir):
        print(f"Cargando datos sintéticos desde {synthetic_img_dir}")
        synthetic_dataset = CloudDataset(
            image_dir=synthetic_img_dir,
            mask_dir=synthetic_mask_dir
        )
        
        # --- 3. COMBINAR DATASETS ---
        print("Combinando dataset original y sintético...")
        train_dataset = ConcatDataset([original_train_dataset, synthetic_dataset])
    else:
        print("No se encontraron datos sintéticos. Usando solo el dataset original.")
        train_dataset = original_train_dataset
        
    print(f"Tamaño final del dataset de entrenamiento: {len(train_dataset)}")

    # --- CREACIÓN DE DATALOADERS (El resto del código no cambia) ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        shuffle=True
    )

    val_dataset = CloudDataset(
        image_dir=Config.VAL_IMG_DIR,
        mask_dir=Config.VAL_MASK_DIR
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        shuffle=False
    )

    # --- LLAMADA A FUNCIÓN DE DISTRIBUCIÓN ELIMINADA ---
    # La función 'imprimir_distribucion_clases_post_augmentation' se elimina
    # porque ya no hay aumentación de datos.

    model = CloudDeepLabV3Plus(num_classes=6).to(Config.DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scaler = GradScaler() 
    best_mIoU = -1.0

    train_miou_history = []
    val_miou_history = []

    for epoch in range(Config.NUM_EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{Config.NUM_EPOCHS} ---")
        train_fn(train_loader, model, optimizer, loss_fn, scaler)
        
        print("Calculando métricas de entrenamiento...")
        train_mIoU, _ = check_metrics(train_loader, model, n_classes=6, device=Config.DEVICE)
        
        print("Calculando métricas de validación...")
        current_mIoU, current_dice = check_metrics(val_loader, model, n_classes=6, device=Config.DEVICE)

        train_miou_history.append(train_mIoU.item())
        val_miou_history.append(current_mIoU.item())

        if current_mIoU > best_mIoU:
            best_mIoU = current_mIoU
            print(f"🔹 Nuevo mejor mIoU: {best_mIoU:.4f} | Dice: {current_dice:.4f}  →  guardando modelo…")
            checkpoint = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_mIoU": best_mIoU,
            }
            torch.save(checkpoint, Config.MODEL_SAVE_PATH)

    save_performance_plot(
        train_history=train_miou_history,
        val_history=val_miou_history,
        save_path=Config.PERFORMANCE_PATH
    )

    print("\nEvaluando el modelo con mejor mIoU guardado…")
    best_model_checkpoint = torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
    model.load_state_dict(best_model_checkpoint['state_dict'])
    best_mIoU, best_dice = check_metrics(val_loader, model, n_classes=6, device=Config.DEVICE)
    print(f"mIoU del modelo guardado: {best_mIoU:.4f} | Dice: {best_dice:.4f}")

if __name__ == "__main__":
    main()