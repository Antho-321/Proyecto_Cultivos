import os
import re
import time
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import generic_filter

from utils import crop_around_classes
from config import Config

def majority_filter(label_map: np.ndarray, window_size: int = 7) -> np.ndarray:
    """Filtro de mayoría sobre un mapa de etiquetas."""
    def _vote(window: np.ndarray) -> int:
        vals, counts = np.unique(window, return_counts=True)
        return vals[np.argmax(counts)]
    return generic_filter(label_map, function=_vote, size=window_size, mode='nearest')

class ValDataset(Dataset):
    _IMG_EXTENSIONS = ('.jpg', '.png', '.jpeg')

    def __init__(self, image_dir: str, mask_dir: str, classes_to_find=None, margin=10):
        self.image_dir = image_dir
        self.mask_dir  = mask_dir
        self.images    = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith(self._IMG_EXTENSIONS)
        ]
        self.classes_to_find = classes_to_find or [1,2,3,4,5]
        self.margin = margin

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img_name   = self.images[idx]
        img_path   = os.path.join(self.image_dir, img_name)
        mask_name  = os.path.splitext(img_name)[0] + "_mask.png"
        mask_path  = os.path.join(self.mask_dir, mask_name)

        image = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        mask  = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask.ndim == 3:
            mask = mask[...,0]

        mask_3d = mask[..., np.newaxis]
        image, mask_3d = crop_around_classes(
            image, mask_3d,
            classes_to_find=self.classes_to_find,
            margin=self.margin
        )
        mask = mask_3d[...,0]

        size = (Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT)
        image = cv2.resize(image, size, interpolation=cv2.INTER_NEAREST)
        mask  = cv2.resize(mask,  size, interpolation=cv2.INTER_NEAREST)

        img_t  = torch.from_numpy(image.astype(np.float32)/255.0).permute(2,0,1)
        mask_t = torch.from_numpy(mask.astype(np.int64))
        return img_t, mask_t

def measure_inference_time(loader: DataLoader,
                           model: torch.nn.Module,
                           device: torch.device,
                           warmup_batches: int = 5) -> float:
    """
    Mide el tiempo medio de inferencia por imagen (en ms).
    """
    model.to(device).eval()
    total_time = 0.0
    total_images = 0

    # Warm-up
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if i >= warmup_batches: break
            _ = model(x.to(device))
            if device.type == 'cuda': torch.cuda.synchronize()

    # Medición
    with torch.no_grad():
        for x, _ in loader:
            batch_size = x.size(0)
            x = x.to(device, non_blocking=True)

            start = time.perf_counter()
            _ = model(x)
            if device.type == 'cuda': torch.cuda.synchronize()
            end = time.perf_counter()

            total_time += (end - start)
            total_images += batch_size

    avg_time_ms = (total_time / total_images) * 1000
    return avg_time_ms

def main():
    torch.backends.cudnn.benchmark = True
    device = torch.device(Config.DEVICE)
    print(f"Usando dispositivo: {device}")

    # Construir DataLoader de validación
    val_loader = DataLoader(
        ValDataset(
            image_dir=Config.VAL_IMG_DIR,
            mask_dir=Config.VAL_MASK_DIR,
            classes_to_find=[1,2,3,4,5],
            margin=10
        ),
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        shuffle=False,
    )

    # Cargar modelo
    model = torch.jit.load(
        "/content/drive/MyDrive/colab/cultivos_deeplab_final.pt",
        map_location=device
    ).eval()

    # Medir tiempo de inferencia
    avg_ms = measure_inference_time(val_loader, model, device)
    print(f"Inferencia promedio: {avg_ms:.2f} ms/imagen")

if __name__ == "__main__":
    main()