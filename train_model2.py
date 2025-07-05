# train.py - Optimized for Tesla T4 GPU

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
from model2 import CloudDeepLabV3Plus
from utils import imprimir_distribucion_clases_post_augmentation, crop_around_classes, save_performance_plot
from config import Config

# =================================================================================
# OPTIMIZED DATASET WITH CACHING AND MEMORY EFFICIENCY
# =================================================================================
class CloudDataset(torch.utils.data.Dataset):
    _IMG_EXTENSIONS = ('.jpg', '.png')

    def __init__(self, image_dir: str, mask_dir: str, transform: A.Compose | None = None, cache_images: bool = False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.cache_images = cache_images
        self.images = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith(self._IMG_EXTENSIONS)
        ]
        
        # Cache for preprocessed images (optional for small datasets)
        self.image_cache = {} if cache_images else None
        self.mask_cache = {} if cache_images else None

    def __len__(self) -> int:
        return len(self.images)

    def _mask_path_from_image_name(self, image_filename: str) -> str:
        name_without_ext = image_filename.rsplit('.', 1)[0]
        mask_filename = f"{name_without_ext}_mask.png"
        return os.path.join(self.mask_dir, mask_filename)

    def __getitem__(self, idx: int):
        img_filename = self.images[idx]
        
        # Check cache first
        if self.cache_images and img_filename in self.image_cache:
            image_cropped = self.image_cache[img_filename]
            mask_cropped = self.mask_cache[img_filename]
        else:
            img_path = os.path.join(self.image_dir, img_filename)
            mask_path = self._mask_path_from_image_name(img_filename)
            
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Máscara no encontrada para {img_filename} en {mask_path}")

            # Load and process images
            image = np.array(Image.open(img_path).convert("RGB"))
            mask = np.array(Image.open(mask_path).convert("L"))

            # Apply cropping before transformations
            mask_3d = np.expand_dims(mask, axis=-1)
            image_cropped, mask_cropped_3d = crop_around_classes(image, mask_3d)
            mask_cropped = mask_cropped_3d.squeeze()
            
            # Cache if enabled
            if self.cache_images:
                self.image_cache[img_filename] = image_cropped.copy()
                self.mask_cache[img_filename] = mask_cropped.copy()

        if self.transform:
            augmented = self.transform(image=image_cropped, mask=mask_cropped)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask

# =================================================================================
# OPTIMIZED TRAINING FUNCTION
# =================================================================================
def train_fn(loader, model, optimizer, loss_fn, scaler, num_classes=6):
    """Optimized training function with reduced metric computation."""
    loop = tqdm(loader, leave=True)
    model.train()

    # Compute metrics less frequently for speed
    compute_metrics_every = max(1, len(loader) // 10)  # Compute every 10% of batches
    
    # Initialize counters
    tp = torch.zeros(num_classes, device=Config.DEVICE)
    fp = torch.zeros(num_classes, device=Config.DEVICE)
    fn = torch.zeros(num_classes, device=Config.DEVICE)
    
    total_loss = 0.0
    batch_count = 0

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(Config.DEVICE, non_blocking=True)
        targets = targets.to(Config.DEVICE, non_blocking=True).long()

        # Forward pass with mixed precision
        with autocast(device_type='cuda', dtype=torch.float16):
            output = model(data)
            predictions = output[0] if isinstance(output, tuple) else output
            loss = loss_fn(predictions, targets)

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        batch_count += 1

        # Compute metrics less frequently
        if batch_idx % compute_metrics_every == 0:
            with torch.no_grad():
                _, predicted_classes = torch.max(predictions, dim=1)
                
                for c in range(num_classes):
                    true_positives = (predicted_classes == c) & (targets == c)
                    false_positives = (predicted_classes == c) & (targets != c)
                    false_negatives = (predicted_classes != c) & (targets == c)

                    tp[c] += true_positives.sum()
                    fp[c] += false_positives.sum()
                    fn[c] += false_negatives.sum()

        # Update progress bar
        loop.set_postfix(loss=loss.item())

    # Calculate final metrics
    epsilon = 1e-6
    iou_per_class = tp / (tp + fp + fn + epsilon)
    dice_per_class = (2 * tp) / (2 * tp + fp + fn + epsilon)
    mean_iou = torch.nanmean(iou_per_class)

    print(f"\nÉpoca de entrenamiento finalizada:")
    print(f"  - Loss promedio: {total_loss / batch_count:.4f}")
    print(f"  - Dice por clase: {dice_per_class.cpu().numpy()}")
    print(f"  - IoU por clase: {iou_per_class.cpu().numpy()}")
    print(f"  - mIoU: {mean_iou:.4f}")
    
    return mean_iou

def check_metrics(loader, model, n_classes: int = 6, device: str = "cuda"):
    """Optimized validation function."""
    eps = 1e-8
    conf_mat = torch.zeros((n_classes, n_classes), dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).long()

            # Use mixed precision for validation too
            with autocast(device_type='cuda', dtype=torch.float16):
                logits = model(x)
                logits = logits[0] if isinstance(logits, tuple) else logits
                preds = torch.argmax(logits, dim=1)

            # Optimized confusion matrix computation
            valid_mask = (y >= 0) & (y < n_classes) & (preds >= 0) & (preds < n_classes)
            y_valid = y[valid_mask]
            preds_valid = preds[valid_mask]
            
            if len(y_valid) > 0:
                indices = y_valid * n_classes + preds_valid
                conf_update = torch.bincount(indices, minlength=n_classes*n_classes).float().view(n_classes, n_classes)
                conf_mat += conf_update

    intersection = torch.diag(conf_mat)
    pred_sum = conf_mat.sum(dim=1)
    true_sum = conf_mat.sum(dim=0)
    union = pred_sum + true_sum - intersection

    iou_per_class = (intersection + eps) / (union + eps)
    dice_per_class = (2 * intersection + eps) / (pred_sum + true_sum + eps)

    miou_macro = iou_per_class.mean()
    dice_macro = dice_per_class.mean()

    print("IoU por clase :", iou_per_class.cpu().numpy())
    print("Dice por clase:", dice_per_class.cpu().numpy())
    print(f"mIoU macro = {miou_macro:.4f} | Dice macro = {dice_macro:.4f}")

    model.train()
    return miou_macro, dice_macro

# =================================================================================
# OPTIMIZED MAIN FUNCTION
# =================================================================================
def main():
    # Enable optimizations for Tesla T4
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    print(f"Using device: {Config.DEVICE}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    
    # Optimized transforms - reduce augmentation intensity for speed
    train_transform = A.Compose([
        A.Resize(height=Config.IMAGE_HEIGHT, width=Config.IMAGE_WIDTH),
        A.Rotate(limit=20, p=0.5),  # Reduced rotation and probability
        A.HorizontalFlip(p=0.5),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(height=Config.IMAGE_HEIGHT, width=Config.IMAGE_WIDTH),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])

    # Create datasets with caching for small datasets
    train_dataset = CloudDataset(
        image_dir=Config.TRAIN_IMG_DIR,
        mask_dir=Config.TRAIN_MASK_DIR,
        transform=train_transform,
        cache_images=False  # Set to True if dataset is small (<1000 images)
    )
    
    # Optimized DataLoader settings for Tesla T4
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        num_workers=min(4, Config.NUM_WORKERS),  # Optimal for Tesla T4
        pin_memory=True,
        shuffle=True,
        persistent_workers=True,  # Keep workers alive between epochs
        prefetch_factor=2,       # Prefetch batches
        drop_last=True          # Ensure consistent batch sizes
    )

    val_dataset = CloudDataset(
        image_dir=Config.VAL_IMG_DIR,
        mask_dir=Config.VAL_MASK_DIR,
        transform=val_transform,
        cache_images=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        num_workers=min(4, Config.NUM_WORKERS),
        pin_memory=True,
        shuffle=False,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=False
    )

    # Skip distribution analysis for faster startup
    # imprimir_distribucion_clases_post_augmentation(train_loader, 6, "...")

    # Model initialization
    model = CloudDeepLabV3Plus(num_classes=6).to(Config.DEVICE)
    print("Compiling the model...")
    model = torch.compile(model, mode='reduce-overhead')  # Better for Tesla T4

    # Optimized loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=Config.LEARNING_RATE,
        weight_decay=1e-4,
        eps=1e-4  # Slightly larger epsilon for stability
    )
    
    # Initialize scaler for mixed precision
    scaler = GradScaler()
    best_mIoU = -1.0

    train_miou_history = []
    val_miou_history = []

    for epoch in range(Config.NUM_EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{Config.NUM_EPOCHS} ---")
        
        # Training
        train_mIoU = train_fn(train_loader, model, optimizer, loss_fn, scaler)
        
        # Validation (less frequent for speed)
        if epoch % 2 == 0 or epoch == Config.NUM_EPOCHS - 1:  # Every 2 epochs
            print("Calculando métricas de validación...")
            current_mIoU, current_dice = check_metrics(val_loader, model, n_classes=6, device=Config.DEVICE)
        else:
            current_mIoU = train_mIoU  # Use training mIoU as approximation
            current_dice = 0.0

        train_miou_history.append(train_mIoU.item())
        val_miou_history.append(current_mIoU.item())

        if current_mIoU > best_mIoU:
            best_mIoU = current_mIoU
            print(f"🔹 Nuevo mejor mIoU: {best_mIoU:.4f} | Dice: {current_dice:.4f} → guardando modelo…")
            checkpoint = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_mIoU": best_mIoU,
            }
            torch.save(checkpoint, Config.MODEL_SAVE_PATH)

    # Save performance plot
    save_performance_plot(
        train_history=train_miou_history,
        val_history=val_miou_history,
        save_path="/content/drive/MyDrive/colab/rendimiento_miou.png"
    )

    print("\nEvaluando el modelo con mejor mIoU guardado…")
    best_model_checkpoint = torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
    model.load_state_dict(best_model_checkpoint['state_dict'])
    best_mIoU, best_dice = check_metrics(val_loader, model, n_classes=6, device=Config.DEVICE)
    print(f"mIoU del modelo guardado: {best_mIoU:.4f} | Dice: {best_dice:.4f}")

if __name__ == "__main__":
    main()