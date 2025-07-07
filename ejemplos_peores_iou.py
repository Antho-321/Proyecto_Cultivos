import os
import re
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from utils import crop_around_classes
from config import Config

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
        img_name  = self.images[idx]
        img_path  = os.path.join(self.image_dir, img_name)
        mask_name = os.path.splitext(img_name)[0] + "_mask.png"
        mask_path = os.path.join(self.mask_dir, mask_name)

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

        # devolvemos también la ruta de la imagen
        return img_t, mask_t, img_path

def check_metrics(loader, model, n_classes: int = 6, device: str = "cuda"):
    eps = 1e-8
    conf_mat = torch.zeros((n_classes, n_classes), dtype=torch.float64, device=device)
    # inicializamos peor IoU por clase (valor alto, ruta vacía)
    worst = {c: (1.0, None) for c in range(n_classes)}
    model.eval()

    with torch.no_grad():
        for x, y, paths in tqdm(loader, desc="Validación"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).long()

            logits = model(x)
            logits = logits[0] if isinstance(logits, (tuple, list)) else logits
            preds = logits.argmax(dim=1)

            # procesamos cada imagen del batch por separado
            for i, img_path in enumerate(paths):
                pi = preds[i]
                yi = y[i]
                flat = (pi * n_classes + yi).view(-1).float()
                conf = torch.histc(
                    flat,
                    bins=n_classes*n_classes,
                    min=0,
                    max=n_classes*n_classes-1
                ).view(n_classes, n_classes)

                # acumulamos para métricas globales
                conf_mat += conf

                # calculamos IoU por clase para esta imagen
                inter = torch.diag(conf)
                ps = conf.sum(dim=1)
                ts = conf.sum(dim=0)
                un = ps + ts - inter
                iou = (inter + eps) / (un + eps)

                # actualizamos peor IoU y ruta
                for c in range(n_classes):
                    if un[c] > eps and iou[c].item() < worst[c][0]:
                        worst[c] = (iou[c].item(), img_path)

    # cálculo de métricas globales
    intersection = torch.diag(conf_mat)
    pred_sum     = conf_mat.sum(dim=1)
    true_sum     = conf_mat.sum(dim=0)
    union        = pred_sum + true_sum - intersection

    iou_per_class  = (intersection + eps) / (union + eps)
    dice_per_class = (2 * intersection + eps) / (pred_sum + true_sum + eps)

    valid = union > eps
    miou_macro = iou_per_class[valid].mean().item()
    dice_macro = dice_per_class[valid].mean().item()

    print("IoU por clase :", iou_per_class.cpu().numpy())
    print("Dice por clase:", dice_per_class.cpu().numpy())
    print(f"mIoU macro (clases presentes) = {miou_macro:.4f} | Dice macro (clases presentes) = {dice_macro:.4f}")

    # impresión de la peor IoU por clase
    for c in range(n_classes):
        iou_val, path = worst[c]
        print(f"Clase {c:2d}: peor IoU = {iou_val:.4f} en {path}")

    return miou_macro, dice_macro

def main():
    torch.backends.cudnn.benchmark = True
    device = torch.device(Config.DEVICE)
    print(f"Usando dispositivo: {device}")

    val_imgs = sorted([
        os.path.join(Config.VAL_IMG_DIR, f)
        for f in os.listdir(Config.VAL_IMG_DIR)
        if re.search(r'\.(jpg|png|jpeg)$', f, re.I)
    ])

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

    model = torch.jit.load(
        "/content/drive/MyDrive/colab/cultivos_deeplab_final.pt",
        map_location=device
    ).eval()

    state = model.state_dict()
    final_w = next(k for k in state if k.endswith('weight'))
    num_classes = state[final_w].shape[0]

    check_metrics(val_loader, model, n_classes=num_classes, device=Config.DEVICE)

if __name__ == "__main__":
    main()
