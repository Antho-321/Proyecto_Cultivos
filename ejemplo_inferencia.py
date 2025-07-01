import os, re, torch, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import Config
from train_test4 import CloudDeepLabV3Plus   # cámbiala si vive en otro módulo

# ─────────────────────────────── 1)  PALETA Y UTILIDADES ──────────────────────
PALETTE = [
    (255,255,255), (128,0,0), (0,128,0),
    (128,128,0),   (0,0,128), (128,0,128)
]
FLAT_PAL = [c for rgb in PALETTE for c in rgb]

def rgb_to_idx(rgb_arr: np.ndarray, palette: list[tuple[int,int,int]]) -> np.ndarray:
    """
    Convierte una máscara RGB (H,W,3) con los mismos colores de `palette`
    a índices 0-len(palette)-1.  Si aparece un color desconocido → 0.
    """
    idx = np.zeros(rgb_arr.shape[:2], dtype=np.uint8)
    for i, color in enumerate(palette):
        mask = np.all(rgb_arr == color, axis=-1)
        idx[mask] = i
    return idx

# ───────────────────────── 2)  ENCONTRAR LA RUTA DE LA GT ─────────────────────
def find_mask(img_path: Path) -> Path | None:
    """
    Busca la máscara correspondiente probando patrones típicos:
        foo.jpg   -> foo_mask.png   (labels/, masks/, misma carpeta…)
        foo.jpeg  -> foo.png
    Devuelve None si no encuentra nada.
    """
    base = re.sub(r"\.(jpg|jpeg)$", "", img_path.name, flags=re.I)
    candidates = [
        img_path.with_name(f"{base}_mask.png"),
        img_path.with_name(f"{base}.png"),
        img_path.parent.parent / "labels" / f"{base}_mask.png",
        img_path.parent.parent / "masks"  / f"{base}_mask.png",
        img_path.parent.parent / "labels" / f"{base}.png",
        img_path.parent.parent / "masks"  / f"{base}.png",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None

def load_gt(mask_path: Path | None, size: tuple[int,int]) -> Image.Image:
    """
    Carga la máscara GT como RGB con la paleta.  
    Si `mask_path` es None → retorna una imagen negra del tamaño `size`.
    """
    if mask_path is None:
        return Image.new("RGB", size, (0,0,0))

    m = Image.open(mask_path)
    if m.mode == "P":                      # ya está paletizada → índices ok
        idx = np.array(m, dtype=np.uint8)
    elif m.mode in ("L", "I"):             # escala de grises (índices)
        idx = np.array(m, dtype=np.uint8)
    else:                                  # RGB → mapear colores a índices
        idx = rgb_to_idx(np.array(m.convert("RGB")), PALETTE)

    gt = Image.fromarray(idx, mode="P")
    gt.putpalette(FLAT_PAL)
    return gt.convert("RGB").resize(size, Image.NEAREST)

# ───────────────────────────── 3)  LISTA DE IMÁGENES ──────────────────────────
BASE_DIR = Path("Balanced/train")
image_paths = [
    BASE_DIR / "images/5-113m3_jpg.rf.1a908ea089918e172ac9b1cfbc81b590.jpg",
    BASE_DIR / "images/101_jpg.rf.2a2a92bdf083fea463b938aa1f3e6bbf.jpg",
    # añade aquí más rutas si quieres…
]

# ───────────────────────────── 4)  MODELO Y TFMS ──────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = torch.load(
    "/content/drive/MyDrive/colab/cultivos_deeplab_final.pt",
    map_location=device, weights_only=False
)
model.eval()

tfm = A.Compose([
    A.Resize(Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH),
    A.Normalize(mean=(0,0,0), std=(1,1,1), max_pixel_value=255.0),
    ToTensorV2(),
])

# ───────────────────────────── 5)  BUCLE PRINCIPAL ────────────────────────────
for idx, img_path in enumerate(image_paths, 1):
    original = Image.open(img_path).convert("RGB")
    gt_mask  = load_gt(find_mask(img_path), original.size)

    # Pre-procesado + inferencia
    tensor = tfm(image=np.array(original))["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        logits = logits[0] if isinstance(logits, tuple) else logits
        pred   = torch.argmax(logits, 1).squeeze().cpu().numpy()

    pred_rgb = Image.fromarray(pred.astype(np.uint8), mode="P")
    pred_rgb.putpalette(FLAT_PAL)
    pred_rgb = pred_rgb.convert("RGB").resize(original.size, Image.NEAREST)

    # Plot & guardar
    fig, ax = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    for a, im, t in zip(
        ax, [original, gt_mask, pred_rgb],
        ["Imagen original", "Máscara GT", "Predicción"]):
        a.imshow(im); a.set_title(t); a.axis("off")

    out_name = f"pred_{idx:02d}_{img_path.stem}.png"
    plt.savefig(out_name, dpi=300, bbox_inches="tight");  plt.close(fig)
    print(f"✔️  Guardado {out_name}")

# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pass
