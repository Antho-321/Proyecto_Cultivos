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
import cv2
# Importa la arquitectura del otro archivo
from model import CloudDeepLabV3Plus 
import matplotlib.pyplot as plt
from distribucion_por_clase   import imprimir_distribucion_clases_post_augmentation
from loss_function            import AsymFocalTverskyLoss
# =================================================================================
# 1. CONFIGURACIÓN
# Centraliza todos los hiperparámetros y rutas aquí.
# =================================================================================
class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- MUY IMPORTANTE: Modifica estas rutas a tus directorios de datos ---
    TRAIN_IMG_DIR = "Balanced/train/images"
    TRAIN_MASK_DIR = "Balanced/train/masks"
    VAL_IMG_DIR = "Balanced/val/images"
    VAL_MASK_DIR = "Balanced/val/masks"
    
    # --- Hiperparámetros de entrenamiento ---
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 8
    NUM_EPOCHS = 200
    NUM_WORKERS = 2
    
    # --- Dimensiones de la imagen ---
    IMAGE_HEIGHT = 256
    IMAGE_WIDTH = 256
    
    # --- Configuraciones adicionales ---
    PIN_MEMORY = True
    LOAD_MODEL = False # Poner a True si quieres continuar un entrenamiento
    MODEL_SAVE_PATH = "best_model.pth.tar"

# =================================================================================
# 2. DATASET PERSONALIZADO
# Clase para cargar las imágenes y sus máscaras de segmentación.
# =================================================================================
# =================================================================================
# 2. DATASET PERSONALIZADO (versión alineada con load_dataset)
# =================================================================================
class CloudDataset(torch.utils.data.Dataset):
    """
    Dataset para segmentación de nubes que asume:
      • Las imágenes están en image_dir y se llaman *.jpg* o *.png*.
      • Las máscaras correspondientes están en mask_dir y se llaman
        '{nombre_imagen_sin_extensión}_mask.png'.
      • Las máscaras son imágenes en escala de grises con 0-255
        (0 = fondo, 255 = nube). Se normalizan a 0-1.
    """
    _IMG_EXTENSIONS = ('.jpg', '.png')

    def __init__(self, image_dir: str, mask_dir: str, transform: A.Compose | None = None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # Lista de archivos que cumplen con las extensiones permitidas
        self.images = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith(self._IMG_EXTENSIONS)
        ]

    def __len__(self) -> int:
        return len(self.images)

    def _mask_path_from_image_name(self, image_filename: str) -> str:
        """
        Convierte 'foto123.jpg' -> 'foto123_mask.png'
        """
        name_without_ext = image_filename.rsplit('.', 1)[0]
        mask_filename = f"{name_without_ext}_mask.png"
        return os.path.join(self.mask_dir, mask_filename)

    def __getitem__(self, idx: int):
        img_filename = self.images[idx]
        img_path = os.path.join(self.image_dir, img_filename)

        mask_path = self._mask_path_from_image_name(img_filename)
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Máscara no encontrada para {img_filename} en {mask_path}")

        # Carga la imagen RGB y la máscara en escala de grises
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]           # Tensor C×H×W, float32
            mask  = augmented["mask"]            # Tensor H×W, int64

        return image, mask

def crop_around_class(
    image: np.ndarray,
    mask: np.ndarray,
    class_id: int = 4,
    margin: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Recorta un rectángulo alrededor de todos los píxeles == class_id en la máscara.

    Args:
        image (H×W×C np.ndarray): Imagen original en RGB o BGR.
        mask  (H×W    np.ndarray): Máscara con valores de clase.
        class_id (int): Valor de la clase a recortar (p.ej. 4).
        margin   (int): Número de píxeles de margen extra alrededor del bounding box.

    Returns:
        cropped_image (h×w×C np.ndarray)
        cropped_mask  (h×w    np.ndarray)

    Si no se encuentra ningún píxel == class_id, devuelve la imagen y máscara originales.
    """
    # Encuentra coordenadas de los píxeles de la clase
    ys, xs = np.where(mask == class_id)
    if ys.size == 0:
        # No hay píxeles de esa clase: devolvemos original
        return image, mask

    # Calcula límites del bounding box
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()

    # Aplica margen sin salirse de los límites
    y0 = max(0, y_min - margin)
    y1 = min(mask.shape[0], y_max + margin + 1)
    x0 = max(0, x_min - margin)
    x1 = min(mask.shape[1], x_max + margin + 1)

    # Recorta
    cropped_image = image[y0:y1, x0:x1]
    cropped_mask  = mask[y0:y1, x0:x1]

    return cropped_image, cropped_mask

class Class4PatchDataset(CloudDataset):
    def __init__(self, *args, patch_size, margin=8, **kwargs):
        super().__init__(*args, **kwargs)
    
        # 2. AHORA, utiliza los argumentos que son exclusivos de ESTA clase.
        #    Estos ya no se pasan al padre.
        self.patch_size = patch_size
        self.margin = margin
        # que contienen la clase 4 en su máscara.
        self.index = []
        for i, rec in enumerate(self.images):  # Suponiendo que 'images' es tu lista de imágenes
            # Asumo que tienes alguna forma de obtener el histograma de clases o de verificar la clase
            mask_path = self._mask_path_from_image_name(rec)
            mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
            hist = np.histogram(mask, bins=np.arange(0, 256))[0]  # Histograma de la máscara
            if hist[4] > 0:  # Verifica si la clase 4 está presente
                self.index.append(i)
                
        self.margin = margin

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        orig_i = self.index[idx]  # Usamos el índice que hemos filtrado

        # 1) Desactivo la transform del padre
        parent_tf = self.transform
        self.transform = None
        img, msk = super().__getitem__(orig_i)  # Aquí NO transforma a tensor
        # 2) Restauro la transform
        self.transform = parent_tf

        # 3) Ahora img y msk son numpy arrays: recorto y redimensiono
        img, msk = crop_around_class(img, msk, class_id=4, margin=self.margin)
        img = cv2.resize(img, (self.patch_size, self.patch_size),
                         interpolation=cv2.INTER_LINEAR)
        msk = cv2.resize(msk, (self.patch_size, self.patch_size),
                         interpolation=cv2.INTER_NEAREST)

        # 4) Finalmente aplico la transform original (que incluye ToTensorV2)
        if parent_tf:
            aug = parent_tf(image=img, mask=msk)
            img, msk = aug["image"], aug["mask"]

        return img, msk

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
        with torch.cuda.amp.autocast(): # Para entrenamiento de precisión mixta
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

def save_performance_plot(train_history, val_history, save_path):
    """
    Guarda un gráfico comparando el mIoU de entrenamiento y validación por época.

    Args:
        train_history (list): Lista con los valores de mIoU de entrenamiento por época.
        val_history (list): Lista con los valores de mIoU de validación por época.
        save_path (str): Ruta donde se guardará el gráfico en formato PNG.
    """
    epochs = range(1, len(train_history) + 1)
    
    plt.style.use('seaborn-v0_8-darkgrid') # Estilo visual atractivo
    fig, ax = plt.subplots(figsize=(12, 7))

    # Graficar ambas curvas
    ax.plot(epochs, train_history, 'o-', color="xkcd:sky blue", label='Entrenamiento (mIoU)', markersize=4)
    ax.plot(epochs, val_history, 'o-', color="xkcd:amber", label='Validación (mIoU)', markersize=4)

    # Títulos y etiquetas
    ax.set_title('Rendimiento del Modelo: mIoU por Época', fontsize=16, weight='bold')
    ax.set_xlabel('Época', fontsize=12)
    ax.set_ylabel('mIoU (Mean Intersection over Union)', fontsize=12)
    
    # Leyenda, cuadrícula y límites
    ax.legend(fontsize=11, frameon=True, shadow=True)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_ylim(0, max(1.0, max(val_history)*1.1)) # Límite Y hasta 1.0 o un poco más del máximo
    ax.set_xticks(epochs) # Asegura que se muestren todas las épocas si no son demasiadas

    plt.tight_layout()
    plt.savefig(save_path, dpi=150) # Guardar con buena resolución
    plt.close(fig) # Liberar memoria
    print(f"📈 Gráfico de rendimiento guardado en '{save_path}'")

# =================================================================================
# 4. FUNCIÓN PRINCIPAL DE EJECUCIÓN
# =================================================================================
def main():
    print(f"Using device: {Config.DEVICE}")
    
    # --- Transformaciones y Aumento de Datos ---

    # Pipeline de validación (sin cambios)
    val_transform = A.Compose([
        A.Resize(height=Config.IMAGE_HEIGHT, width=Config.IMAGE_WIDTH),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])

    # <<< CAMBIO PRINCIPAL: Pipeline de aumentos más agresivo para Clase 4 >>>
    # Este pipeline incluye las transformaciones que pediste (rotación, traslación) y más.
    train_class4_transform = A.Compose([
        A.Resize(height=Config.IMAGE_HEIGHT, width=Config.IMAGE_WIDTH),
        
        # 1. Transformación geométrica más potente
        A.ShiftScaleRotate(
            shift_limit=0.1,         # Traslación de hasta un 10%
            scale_limit=0.15,        # Zoom de hasta un 15%
            rotate_limit=60,         # Rotación de hasta 60 grados
            p=0.9,
            border_mode=cv2.BORDER_CONSTANT,
            mask_value=0 # Rellena los bordes de la máscara con la clase 0 (fondo)
        ),
        
        # 2. Volteos
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        
        # 3. Aumentos de color y brillo (ayuda a la robustez)
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
        A.HueSaturationValue(p=0.5),

        # 4. Dropout espacial (fuerza al modelo a aprender de partes de la imagen)
        A.CoarseDropout(
            max_holes=8,
            max_height=32,
            max_width=32,
            min_holes=1,
            fill_value=0, # Rellena los huecos con la clase 0
            mask_fill_value=0, # Rellena huecos en la máscara con clase 0
            p=0.5
        ),

        # 5. Normalización y conversión a Tensor (siempre al final)
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])

    # --- Creación de Datasets y DataLoaders ---
    train_dataset = Class4PatchDataset(
        image_dir=Config.TRAIN_IMG_DIR,
        mask_dir=Config.TRAIN_MASK_DIR,
        transform=train_class4_transform, # <<< CAMBIO: Usamos el nuevo pipeline agresivo aquí
        margin=8,
        patch_size=Config.IMAGE_HEIGHT
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        shuffle=True,
    )

    imprimir_distribucion_clases_post_augmentation(train_loader, 6,
        "Distribución de clases en ENTRENAMIENTO (post-aug)")

    val_dataset = Class4PatchDataset(
      image_dir=Config.VAL_IMG_DIR,
      mask_dir=Config.VAL_MASK_DIR,
      transform=val_transform, # ¡OJO! Usa el val_transform (sin aumentos aleatorios)
      margin=8,
      patch_size=Config.IMAGE_HEIGHT
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        shuffle=False
    )
    
    # --- El resto del código de instanciación y bucle de entrenamiento no necesita cambios ---
    model = CloudDeepLabV3Plus(num_classes=6).to(Config.DEVICE)
    loss_fn = AsymFocalTverskyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scaler = GradScaler() 
    best_mIoU = -1.0
    train_miou_history = []
    val_miou_history = []

    # --- Bucle Principal de Entrenamiento (sin cambios) ---
    for epoch in range(Config.NUM_EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{Config.NUM_EPOCHS} ---")
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        print("\n--- Métricas de Entrenamiento ---")
        train_miou, _ = check_metrics(train_loader, model, device=Config.DEVICE)
        print("\n--- Métricas de Validación ---")
        val_miou, current_dice = check_metrics(val_loader, model, device=Config.DEVICE)
        
        train_miou_history.append(train_miou.cpu().item())
        val_miou_history.append(val_miou.cpu().item())
        
        if val_miou > best_mIoU:
            best_mIoU = val_miou
            print(f"🔹 Nuevo mejor mIoU: {best_mIoU:.4f} | Dice: {current_dice:.4f}  →  guardando modelo…")
            checkpoint = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_mIoU": best_mIoU,
            }
            torch.save(checkpoint, Config.MODEL_SAVE_PATH)

    # --- Evaluación final y guardado de gráfico (sin cambios) ---
    print("\nEvaluando el modelo con mejor mIoU guardado…")
    # Cargar el mejor modelo antes de la evaluación final
    # (Este paso es importante para asegurar que evalúas el mejor checkpoint)
    checkpoint = torch.load(Config.MODEL_SAVE_PATH)
    model.load_state_dict(checkpoint['state_dict'])
    
    best_mIoU, best_dice = check_metrics(val_loader, model, n_classes=6, device=Config.DEVICE)
    print(f"mIoU del modelo guardado: {best_mIoU:.4f} | Dice: {best_dice:.4f}")
    save_performance_plot(train_miou_history, val_miou_history, "/content/drive/MyDrive/colab/rendimiento_miou.png")

if __name__ == "__main__":
    main()