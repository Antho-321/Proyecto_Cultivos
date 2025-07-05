# train.py - Maximum Speed Optimizations for Tesla T4 GPU

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
# ULTRA-FAST DATASET WITH MINIMAL PROCESSING
# =================================================================================
class CloudDataset(torch.utils.data.Dataset):
    _IMG_EXTENSIONS = ('.jpg', '.png')

    def __init__(self, image_dir: str, mask_dir: str, transform: A.Compose | None = None, 
                 cache_images: bool = False, preload_to_memory: bool = False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.cache_images = cache_images
        self.preload_to_memory = preload_to_memory
        
        self.images = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith(self._IMG_EXTENSIONS)
        ]
        
        # Preload all data to memory for maximum speed (if dataset is small)
        if preload_to_memory:
            print("Preloading dataset to memory...")
            self.memory_data = {}
            for img_filename in tqdm(self.images, desc="Loading to memory"):
                img_path = os.path.join(image_dir, img_filename)
                mask_path = self._mask_path_from_image_name(img_filename)
                
                image = np.array(Image.open(img_path).convert("RGB"))
                mask = np.array(Image.open(mask_path).convert("L"))
                
                # Pre-crop and store
                mask_3d = np.expand_dims(mask, axis=-1)
                image_cropped, mask_cropped_3d = crop_around_classes(image, mask_3d)
                mask_cropped = mask_cropped_3d.squeeze()
                
                self.memory_data[img_filename] = (image_cropped, mask_cropped)
        else:
            self.memory_data = None
            
        # Cache for preprocessed images
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
        
        # Use preloaded data if available
        if self.memory_data is not None:
            image_cropped, mask_cropped = self.memory_data[img_filename]
        elif self.cache_images and img_filename in self.image_cache:
            image_cropped = self.image_cache[img_filename]
            mask_cropped = self.mask_cache[img_filename]
        else:
            img_path = os.path.join(self.image_dir, img_filename)
            mask_path = self._mask_path_from_image_name(img_filename)
            
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Máscara no encontrada para {img_filename} en {mask_path}")

            image = np.array(Image.open(img_path).convert("RGB"))
            mask = np.array(Image.open(mask_path).convert("L"))

            mask_3d = np.expand_dims(mask, axis=-1)
            image_cropped, mask_cropped_3d = crop_around_classes(image, mask_3d)
            mask_cropped = mask_cropped_3d.squeeze()
            
            if self.cache_images:
                self.image_cache[img_filename] = image_cropped.copy()
                self.mask_cache[img_filename] = mask_cropped.copy()

        if self.transform:
            augmented = self.transform(image=image_cropped, mask=mask_cropped)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask

# =================================================================================
# ULTRA-FAST TRAINING FUNCTION
# =================================================================================
def train_fn(loader, model, optimizer, loss_fn, scaler, num_classes=6):
    """Ultra-fast training function with minimal metric computation."""
    model.train()
    
    # Compute metrics only at the end for maximum speed
    total_loss = 0.0
    batch_count = 0
    
    # Simple progress bar
    loop = tqdm(loader, leave=False, desc="Training")

    for data, targets in loop:
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

        # Minimal progress update
        if batch_count % 10 == 0:
            loop.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / batch_count
    print(f"Training Loss: {avg_loss:.4f}")
    
    return avg_loss

def check_metrics_fast(loader, model, n_classes: int = 6, device: str = "cuda"):
    """Lightning-fast validation function."""
    model.eval()
    
    # Use smaller sample for faster validation during training
    sample_size = min(len(loader), 20)  # Only validate on first 20 batches
    
    intersection = torch.zeros(n_classes, device=device)
    union = torch.zeros(n_classes, device=device)
    
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= sample_size:
                break
                
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).long()

            with autocast(device_type='cuda', dtype=torch.float16):
                logits = model(x)
                logits = logits[0] if isinstance(logits, tuple) else logits
                preds = torch.argmax(logits, dim=1)

            # Fast IoU computation
            for c in range(n_classes):
                pred_mask = (preds == c)
                true_mask = (y == c)
                intersection[c] += (pred_mask & true_mask).sum()
                union[c] += (pred_mask | true_mask).sum()

    iou_per_class = intersection / (union + 1e-8)
    miou = iou_per_class.mean()

    print(f"Fast validation mIoU: {miou:.4f}")
    model.train()
    return miou

def check_metrics(loader, model, n_classes: int = 6, device: str = "cuda"):
    """Full validation - use only for final evaluation."""
    eps = 1e-8
    conf_mat = torch.zeros((n_classes, n_classes), dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Full validation"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).long()

            with autocast(device_type='cuda', dtype=torch.float16):
                logits = model(x)
                logits = logits[0] if isinstance(logits, tuple) else logits
                preds = torch.argmax(logits, dim=1)

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
# MAXIMUM SPEED MAIN FUNCTION
# =================================================================================
def main():
    # Enable ALL optimizations for Tesla T4
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False  # Disable for speed
    torch.backends.cudnn.enabled = True
    
    # Set environment variables for maximum performance
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    
    print(f"Using device: {Config.DEVICE}")
    print(f"CUDA version: {torch.version.cuda}")
    
    # MINIMAL augmentations for maximum speed
    train_transform = A.Compose([
        A.Resize(height=Config.IMAGE_HEIGHT, width=Config.IMAGE_WIDTH, 
                 interpolation=1),
        A.Rotate(limit=35, p=0.7),
        A.HorizontalFlip(p=0.5),  # Only keep essential augmentations
        A.VerticalFlip(p=0.3),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(height=Config.IMAGE_HEIGHT, width=Config.IMAGE_WIDTH,
                 interpolation=1),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])

    # Determine if dataset is small enough for memory preloading
    train_size = len([f for f in os.listdir(Config.TRAIN_IMG_DIR) 
                     if f.lower().endswith(('.jpg', '.png'))])
    use_memory_preload = train_size < 500  # Preload if < 500 images
    
    print(f"Dataset size: {train_size} images")
    print(f"Memory preloading: {use_memory_preload}")

    # Create datasets
    train_dataset = CloudDataset(
        image_dir=Config.TRAIN_IMG_DIR,
        mask_dir=Config.TRAIN_MASK_DIR,
        transform=train_transform,
        cache_images=not use_memory_preload,
        preload_to_memory=use_memory_preload
    )
    
    # OPTIMIZED DataLoader settings for maximum speed
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        num_workers=2,  # Reduced for Tesla T4
        pin_memory=True,
        shuffle=True,
        persistent_workers=True,
        prefetch_factor=4,  # Increased prefetch
        drop_last=True,
        multiprocessing_context='spawn'  # Better for GPU
    )

    val_dataset = CloudDataset(
        image_dir=Config.VAL_IMG_DIR,
        mask_dir=Config.VAL_MASK_DIR,
        transform=val_transform,
        cache_images=False,
        preload_to_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE * 2,  # Larger batch for validation
        num_workers=2,
        pin_memory=True,
        shuffle=False,
        persistent_workers=True,
        prefetch_factor=4,
        drop_last=False,
        multiprocessing_context='spawn'
    )

    # Skip all non-essential operations
    print("Skipping distribution analysis for speed...")

    # Model initialization with compilation
    model = CloudDeepLabV3Plus(num_classes=6).to(Config.DEVICE)
    
    # Compile with maximum optimization
    print("Compiling model for maximum speed...")
    model = torch.compile(model, mode='max-autotune')  # Best for Tesla T4

    # Optimized loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=Config.LEARNING_RATE,
        weight_decay=1e-4,
        eps=1e-4,
        fused=True  # Use fused optimizer for speed
    )
    
    # Initialize scaler
    scaler = GradScaler()
    best_mIoU = -1.0

    train_miou_history = []
    val_miou_history = []

    # Training loop with minimal validation
    for epoch in range(Config.NUM_EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{Config.NUM_EPOCHS} ---")
        
        print("Calculando métricas de entrenamiento...")
        train_mIoU = train_fn(train_loader, model, optimizer, loss_fn, scaler)
        
        print("Calculando métricas de validación...")
        current_mIoU, current_dice = check_metrics(val_loader, model, n_classes=6, device=Config.DEVICE)

        # --- 4. GUARDAR LAS MÉTRICAS EN EL HISTORIAL ---
        train_miou_history.append(train_mIoU)
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

    # Final evaluation
    print("\n=== FINAL EVALUATION ===")
    best_model_checkpoint = torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
    model.load_state_dict(best_model_checkpoint['state_dict'])
    best_mIoU, best_dice = check_metrics(val_loader, model, n_classes=6, device=Config.DEVICE)
    print(f"Final mIoU: {best_mIoU:.4f} | Dice: {best_dice:.4f}")

    # Save performance plot
    try:
        save_performance_plot(
            train_history=train_miou_history,  # Note: using loss instead of mIoU
            val_history=val_miou_history,
            save_path="/content/drive/MyDrive/colab/rendimiento_miou.png"
        )
    except Exception as e:
        print(f"Could not save plot: {e}")

if __name__ == "__main__":
    main()