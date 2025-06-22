# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
from PIL import Image
import numpy as np
import cupy as cp
# Importa la arquitectura del otro archivo
from model import CloudDeepLabV3Plus
import matplotlib.pyplot as plt
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

        # 1. Load data as NumPy arrays on the CPU
        image_np = np.array(Image.open(img_path).convert("RGB"))
        mask_np = np.array(Image.open(mask_path).convert("L"))
        mask_3d_np = np.expand_dims(mask_np, axis=-1)

        # 2. Move data from CPU (NumPy) to GPU (CuPy)
        image_gpu = cp.asarray(image_np)
        mask_3d_gpu = cp.asarray(mask_3d_np)
        
        # 3. Apply the GPU-based cropping function
        image_cropped_gpu, mask_cropped_3d_gpu = crop_around_classes(image_gpu, mask_3d_gpu)

        # 4. Move the cropped result from GPU (CuPy) back to CPU (NumPy)
        #    This is crucial because Albumentations and PyTorch's collate_fn
        #    expect CPU-based NumPy arrays or PIL Images.
        image_cropped_np = cp.asnumpy(image_cropped_gpu)
        mask_cropped_3d_np = cp.asnumpy(mask_cropped_3d_gpu)

        # 5. Squeeze the mask for Albumentations
        mask_cropped_np = mask_cropped_3d_np.squeeze()

        if self.transform:
            # Pass the CROPPED NUMPY arrays to the transformations
            augmented = self.transform(image=image_cropped_np, mask=mask_cropped_np)
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

        with autocast(device_type=Config.DEVICE, dtype=torch.float16):
            # Desempaquetamos o seleccionamos la salida principal
            output = model(data)
            predictions = output[0] if isinstance(output, tuple) else output
            loss = loss_fn(predictions, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())

def check_metrics(loader, model, n_classes=6, device="cuda"):
    """
    Devuelve mIoU macro y Dice macro de forma vectorizada y eficiente.
    Calcula una matriz de confusión para todo el dataset y luego deriva las métricas.
    """
    model.eval()
    eps = 1e-8
    
    # Matriz de confusión acumulada para todo el dataset.
    # Filas: Clases verdaderas (Ground Truth)
    # Columnas: Clases predichas (Predictions)
    confusion_matrix = torch.zeros((n_classes, n_classes), dtype=torch.int64, device=device)

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).long() # Shape: (N, H, W)

            # Obtener predicciones del modelo
            output = model(x)
            logits = output[0] if isinstance(output, tuple) else output
            preds = torch.argmax(logits, dim=1) # Shape: (N, H, W)

            # Aplanar las máscaras para facilitar el cálculo
            y_flat = y.flatten()       # Shape: (N*H*W,)
            preds_flat = preds.flatten() # Shape: (N*H*W,)

            # Crear una máscara para ignorar los píxeles que no nos interesan (si los hubiera)
            # En este caso, consideramos todas las clases de 0 a n_classes-1
            mask = (y_flat >= 0) & (y_flat < n_classes)
            y_valid = y_flat[mask]
            preds_valid = preds_flat[mask]

            # El truco para actualizar la matriz de confusión de forma vectorizada
            # Cada par (y_true, y_pred) corresponde a una celda única en la matriz.
            indices = y_valid * n_classes + preds_valid
            
            # torch.bincount cuenta las ocurrencias de cada índice.
            # Esto nos da los valores para la matriz de confusión de este batch.
            batch_cm = torch.bincount(indices, minlength=n_classes**2).reshape(n_classes, n_classes)
            
            confusion_matrix += batch_cm

    # --- Ahora, calcular todas las métricas a partir de la matriz de confusión final ---
    
    # True Positives (TP) son los elementos de la diagonal
    intersection = torch.diag(confusion_matrix)

    # False Positives (FP) son la suma de cada columna, menos los TP
    # False Negatives (FN) son la suma de cada fila, menos los TP
    pred_sum = confusion_matrix.sum(dim=0) # Suma de predicciones por clase
    true_sum = confusion_matrix.sum(dim=1) # Suma de etiquetas reales por clase

    # Cálculo de mIoU
    union = pred_sum + true_sum - intersection
    iou_per_class = (intersection + eps) / (union + eps)
    miou_macro = iou_per_class.mean()

    # Cálculo de Dice
    dice_num = 2 * intersection
    dice_den = pred_sum + true_sum
    dice_per_class = (dice_num + eps) / (dice_den + eps)
    dice_macro = dice_per_class.mean()

    # Imprimir resultados
    print("IoU por clase :", iou_per_class.cpu().numpy())
    print("Dice por clase:", dice_per_class.cpu().numpy())
    print(f"mIoU macro = {miou_macro:.4f} | Dice macro = {dice_macro:.4f}")

    model.train()
    return miou_macro, dice_macro

def worker_init_fn(worker_id):
    """Initializes the CUDA device for CuPy in each worker process."""
    # The worker gets a new process, so we need to set the device for CuPy
    # We use the device PyTorch is using in the main process
    try:
        gpu_id = torch.cuda.current_device()
        cp.cuda.Device(gpu_id).use()
    except Exception as e:
        print(f"Error initializing CuPy in worker {worker_id}: {e}")

# =================================================================================
# 4. FUNCIÓN PRINCIPAL DE EJECUCIÓN (Sin cambios)
# =================================================================================
def main():
    torch.backends.cudnn.benchmark = True

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
        shuffle=True,
        worker_init_fn=worker_init_fn
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
        shuffle=False,
        worker_init_fn=worker_init_fn
    )

    imprimir_distribucion_clases_post_augmentation(train_loader, 6,
        "Distribución de clases en ENTRENAMIENTO (post-aug)")

    model = CloudDeepLabV3Plus(num_classes=6).to(Config.DEVICE)
    print("Compiling the model... (this may take a minute)")
    model = torch.compile(model)
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
    # CRITICAL: Set the start method to 'spawn'
    # This must be done inside the __name__ == "__main__" block
    # and before any CUDA operations or DataLoader instantiation.
    try:
        mp.set_start_method('spawn', force=True)
        print("Multiprocessing start method set to 'spawn'.")
    except RuntimeError:
        pass # It may already be set

    main()