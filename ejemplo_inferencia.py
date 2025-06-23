import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import Config
from train_test4 import CloudDeepLabV3Plus  # Ajusta la importación si la clase está en otro módulo

def main():
    # 1) Rutas
    image_path = os.path.join(Config.VAL_IMG_DIR,
                              "5-365m3_jpg.rf.03efe0cdb37570e749c0c0c55be70e12.jpg")
    mask_path  = os.path.join(Config.VAL_MASK_DIR,
                              "5-365m3_jpg.rf.03efe0cdb37570e749c0c0c55be70e12.png")
    model_path = "/content/drive/MyDrive/colab/cultivos_deeplab_final.pt"

    # 2) Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paleta de colores definida antes para ser usada por GT y predicción
    palette = [
        (  0,   0,   0),  # clase 0: negro
        (128,   0,   0),  # clase 1: marrón
        (  0, 128,   0),  # clase 2: verde
        (128, 128,   0),  # clase 3: oliva
        (  0,   0, 128),  # clase 4: azul
        (128,   0, 128),  # clase 5: púrpura
    ]
    flat = [c for rgb in palette for c in rgb]

    # 3) Carga del modelo
    # El archivo .pt es un modelo TorchScript, así que lo cargamos directamente.
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()

    # 4) Carga de la imagen y la máscara GT
    original = Image.open(image_path).convert("RGB")
    try:
        # 4a) Cargar la GT como índice 0–5
        gt_idx = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
    except FileNotFoundError:
        # Placeholder negro
        gt_idx = np.zeros((original.height, original.width), dtype=np.uint8)

    # 4b) Convierto a 'P' y le pongo paleta
    gt_mask = Image.fromarray(gt_idx, mode="P")
    gt_mask.putpalette(flat)
    gt_mask = gt_mask.convert("RGB").resize(
        original.size, resample=Image.NEAREST
    )

    # 5) Pre-procesado para el modelo
    transform = A.Compose([
        A.Resize(Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH),
        A.Normalize(mean=[0,0,0], std=[1,1,1], max_pixel_value=255.0),
        ToTensorV2(),
    ])
    img_np = np.array(original)
    aug    = transform(image=img_np)
    x      = aug["image"].unsqueeze(0).to(device)  # (1,C,H,W)

    # 6) Inferencia y obtención de máscara predicha
    with torch.no_grad():
        out = model(x)
        out = out[0] if isinstance(out, tuple) else out
        pred = torch.argmax(out, dim=1).squeeze().cpu().numpy()  # (H,W) índices 0–5

    # 7) Convertir índice→RGB usando la paleta
    # Crear imagen paletizada para la predicción
    pred_img = Image.fromarray(pred.astype(np.uint8), mode="P")
    pred_img.putpalette(flat)
    # Pasar a RGB y redimensionar al tamaño original
    pred_rgb = pred_img.convert("RGB").resize(original.size, resample=Image.NEAREST)

    # 8) Plot de los tres paneles
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    axes[0].imshow(original)
    axes[0].set_title("Imagen Original")
    axes[0].axis("off")

    axes[1].imshow(gt_mask)  # Ahora GT usa la misma paleta
    axes[1].set_title("Máscara GT")
    axes[1].axis("off")

    axes[2].imshow(pred_rgb)
    axes[2].set_title("Máscara Predicha")
    axes[2].axis("off")

    # Guardamos sin recortar nada
    plt.savefig("image_and_masks.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()