import os, re, torch, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import Config
from train_test4 import CloudDeepLabV3Plus   # cambia si la clase está en otro módulo

# ──────────────────────────────────────────────────────────────────────────────
# 1)  LISTA DE RUTAS A PROCESAR  (puedes añadir o quitar libremente)
BASE_DIR = Path("Balanced/train")
image_paths = [
    BASE_DIR / "images/5-113m3_jpg.rf.1a908ea089918e172ac9b1cfbc81b590.jpg",  # clase 1
    BASE_DIR / "images/101_jpg.rf.2a2a92bdf083fea463b938aa1f3e6bbf.jpg",      # clase 2
    BASE_DIR / "images/101_jpg.rf.2a2a92bdf083fea463b938aa1f3e6bbf.jpg",      # clase 3
    BASE_DIR / "images/101_jpg.rf.2a2a92bdf083fea463b938aa1f3e6bbf.jpg",      # clase 4
    BASE_DIR / "images/101_jpg.rf.2a2a92bdf083fea463b938aa1f3e6bbf.jpg",      # clase 5
]

def mask_from_img(img_path: Path) -> Path:
    """
    Construye la ruta de la máscara a partir del nombre de la imagen.
    Ejemplo: images/foo.jpg -> labels/foo_mask.png
    """
    name = re.sub(r"\.jpg$", "_mask.png", img_path.name, flags=re.I)
    return img_path.with_name(name).with_suffix(".png").parent.parent / "labels" / name

# ──────────────────────────────────────────────────────────────────────────────
# 2)  SET-UP
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model   = torch.load(
    "/content/drive/MyDrive/colab/cultivos_deeplab_final.pt",
    map_location=device, weights_only=False
)
model.eval()

palette = [
    (255,255,255), (128,0,0), (0,128,0),
    (128,128,0),   (0,0,128), (128,0,128)
]
flat_pal = [c for rgb in palette for c in rgb]

tfm = A.Compose([
    A.Resize(Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH),
    A.Normalize(mean=(0,0,0), std=(1,1,1), max_pixel_value=255.0),
    ToTensorV2(),
])

# ──────────────────────────────────────────────────────────────────────────────
# 3)  BUCLE PRINCIPAL
for idx, img_path in enumerate(image_paths, start=1):
    mask_path = mask_from_img(img_path)

    # 3a) Imagen y (opcional) máscara GT
    original = Image.open(img_path).convert("RGB")
    if mask_path.exists():
        gt_idx = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
    else:
        gt_idx = np.zeros((original.height, original.width), dtype=np.uint8)

    gt_rgb = Image.fromarray(gt_idx, mode="P")
    gt_rgb.putpalette(flat_pal)
    gt_rgb = gt_rgb.convert("RGB").resize(original.size, Image.NEAREST)

    # 3b) Pre-procesado + inferencia
    x = tfm(image=np.array(original))["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        logits = logits[0] if isinstance(logits, tuple) else logits
        pred   = torch.argmax(logits, dim=1).squeeze().cpu().numpy()

    pred_rgb = Image.fromarray(pred.astype(np.uint8), mode="P")
    pred_rgb.putpalette(flat_pal)
    pred_rgb = pred_rgb.convert("RGB").resize(original.size, Image.NEAREST)

    # 3c) Plot + guardado
    fig, ax = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    for a, img, title in zip(ax,
        [original, gt_rgb, pred_rgb],
        ["Imagen original", "GT", "Predicción"]):
        a.imshow(img); a.set_title(title); a.axis("off")

    out_name = f"pred_{idx:02d}_{img_path.stem}.png"
    plt.savefig(out_name, dpi=300, bbox_inches="tight")
    plt.close(fig)          # libera memoria
    print(f"Guardado {out_name}")

# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pass    # el script ya corre todo al importarse; pon aquí lógica extra si quieres
