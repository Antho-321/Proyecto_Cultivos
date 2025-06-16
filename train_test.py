# train.py  ── versión revisada 2025-06-15
import os, cv2, torch, albumentations as A
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from model                    import CloudDeepLabV3Plus
from loss_function            import AsymFocalTverskyLoss
from distribucion_por_clase   import imprimir_distribucion_clases_post_augmentation
# ──────────────────────────────────────────────────────────────────────────────
# 1 ▸ CONFIGURACIÓN
# ──────────────────────────────────────────────────────────────────────────────
class Config:
    DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
    TRAIN_IMG_DIR, TRAIN_MASK_DIR = "Balanced/train/images", "Balanced/train/masks"
    VAL_IMG_DIR,   VAL_MASK_DIR   = "Balanced/val/images",   "Balanced/val/masks"

    LEARNING_RATE = 1e-4
    BATCH_SIZE    = 8
    NUM_EPOCHS    = 200
    NUM_WORKERS   = 2

    IMAGE_HEIGHT  = 256
    IMAGE_WIDTH   = 256

    # ajustes críticos
    USE_AMP       = False              # ← ❶  FP16 desactivado hasta estabilizar
    MAX_W_CLS     = 3.0                # ← ❷  cota para pesos de clase
    EPS_TVERSKY   = 1e-4               # ← ❸  ε más grande dentro del loss
    CLIP_NORM     = 1.0                # ← ❹  grad-clipping

    PIN_MEMORY    = True
    MODEL_SAVE_PATH = "best_model.pth.tar"
# ──────────────────────────────────────────────────────────────────────────────
# 2 ▸ DATASET  (sin cambios relevantes)
# ──────────────────────────────────────────────────────────────────────────────
class CloudPatchDatasetBalanced(torch.utils.data.Dataset):
    _IMG_EXT = ('.jpg', '.png')

    def __init__(self, image_dir, mask_dir, patch_size=128, stride=128,
                 min_fg_ratio=0.4, transform=None):
        self.image_dir, self.mask_dir = image_dir, mask_dir
        self.patch_size, self.stride  = patch_size, stride
        self.min_fg_ratio, self.transform = min_fg_ratio, transform

        self.images = [f for f in os.listdir(image_dir)
                       if f.lower().endswith(self._IMG_EXT)]
        self.index  = []
        for img_idx, img_name in enumerate(self.images):
            mask_np = np.array(
                Image.open(os.path.join(mask_dir,
                          f"{img_name.rsplit('.',1)[0]}_mask.png"))
                .convert("L"), dtype=np.uint8)

            H, W = mask_np.shape
            for y in range(0, H-patch_size+1, stride):
                for x in range(0, W-patch_size+1, stride):
                    patch = mask_np[y:y+patch_size, x:x+patch_size]
                    if (patch != 0).mean() < self.min_fg_ratio: 
                        continue
                    hist = np.bincount(patch.flatten(), minlength=6)
                    self.index.append(dict(img_idx=img_idx,y=y,x=x,hist=hist))

    def __len__(self):               return len(self.index)
    def _load(self, folder, name):   return np.array(Image.open(
                                    os.path.join(folder, name)).convert("RGB"))
    def __getitem__(self, i):
        r   = self.index[i]; name = self.images[r['img_idx']]
        img = self._load(self.image_dir, name)[r['y']:r['y']+self.patch_size,
                                               r['x']:r['x']+self.patch_size]
        msk = np.array(Image.open(os.path.join(
                     self.mask_dir,f"{name.rsplit('.',1)[0]}_mask.png"))
                     .convert("L"),dtype=np.uint8)[r['y']:r['y']+self.patch_size,
                                                   r['x']:r['x']+self.patch_size]
        if self.transform:
            aug = self.transform(image=img, mask=msk); img, msk = aug["image"], aug["mask"]
        return img, msk
# ──────────────────────────────────────────────────────────────────────────────
# 3 ▸ SAMPLER  (igual que antes)
# ──────────────────────────────────────────────────────────────────────────────
class PixelBalancedSampler(torch.utils.data.Sampler):
    def __init__(self, ds, target_pixels_per_class, batch_size, drop_last=True):
        self.ds, self.tpp = ds, np.asarray(target_pixels_per_class)
        self.B, self.drop_last = batch_size, drop_last
        self.patch_hist = [rec["hist"] for rec in ds.index]

    def __iter__(self):
        order = np.random.permutation(len(self.ds))
        batch, cum = [], np.zeros(6, np.int64)
        for idx in order:
            h = self.patch_hist[idx]
            if (cum + h <= self.tpp).all() or not batch:
                batch.append(idx); cum += h
                if len(batch) == self.B: yield batch; batch, cum = [], np.zeros(6, np.int64)
        if batch and not self.drop_last: yield batch
    def __len__(self): return len(self.ds)//self.B
# ──────────────────────────────────────────────────────────────────────────────
# 4 ▸ UTILIDADES
# ──────────────────────────────────────────────────────────────────────────────
def compute_class_weights(ds, n_classes=6, method="median"):
    pix, tot = np.zeros(n_classes), np.zeros_like(np.zeros(n_classes))
    for _, m in ds:
        m = np.array(m); t = m.size
        for c in range(n_classes):
            pc = (m==c).sum(); pix[c]+=pc; 
            if pc>0: tot[c]+=t
    if method=="inverse":
        w = pix.sum()/(pix+1e-8)
    else:                      # MFB
        freq = pix/(tot+1e-8); med = np.median(freq[freq>0]); w = med/(freq+1e-8)
    w[0]=0.0                             # fondo
    return np.clip(w, 0.0, Config.MAX_W_CLS)  # ← ❷  limitamos
# ──────────────────────────────────────────────────────────────────────────────
def check_metrics(loader, model, n_classes=6, device="cuda", ignore_background=True):
    eps = 1e-8
    inter, union, d_num, d_den = (torch.zeros(n_classes, dtype=torch.float64, device=device)
                                  for _ in range(4))
    model.eval()
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device).long()
            p   = torch.argmax(model(x),1)
            for c in range(n_classes):
                pc, tc = p==c, y==c
                i      = (pc&tc).sum().double()
                u      = pc.sum().double()+tc.sum().double()-i
                if u==0:                 # ← ❺  clase ausente → ignoramos
                    continue
                inter[c]+=i; union[c]+=u
                d_num[c]+=2*i; d_den[c]+=pc.sum()+tc.sum()
    valid = slice(1,None) if ignore_background else slice(0,None)
    iou   = (inter+eps)/(union+eps);  dice = (d_num+eps)/(d_den+eps)
    vals_iou  = iou.cpu().numpy()
    vals_dice = dice.cpu().numpy()

    # 1) Con comprensión de listas + f-string
    print("IoU :", [f"{v:.4f}" for v in vals_iou])
    print("Dice:", [f"{v:.4f}" for v in vals_dice])
    return iou[valid].mean(), dice[valid].mean()
# ──────────────────────────────────────────────────────────────────────────────
def train_one_epoch(loader, model, loss_fn, optimizer, scaler):
    model.train()
    loop = tqdm(loader, leave=True)
    for x,y in loop:
        x = x.to(Config.DEVICE); y = y.to(Config.DEVICE).long()
        with autocast(enabled=Config.USE_AMP, device_type="cuda", dtype=torch.float16):
            pred = model(x); loss = loss_fn(pred, y)
        if not torch.isfinite(loss):             # ← ❻  early-stop batch
            print("⚠️  pérdida no finita, se descarta el lote"); optimizer.zero_grad(); continue
        optimizer.zero_grad()
        if Config.USE_AMP:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.CLIP_NORM)  # ← ❹
            scaler.step(optimizer); scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.CLIP_NORM)
            optimizer.step()
        loop.set_postfix(loss=float(loss))
# ──────────────────────────────────────────────────────────────────────────────
# 5 ▸ FUNCIÓN PRINCIPAL
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print(f"🖥  Device → {Config.DEVICE}")
    # Aumentos
    train_tf = A.Compose([
        A.RandomResizedCrop((Config.IMAGE_HEIGHT,Config.IMAGE_WIDTH),scale=(0.5,1),ratio=(0.75,1.33),
                            interpolation=cv2.INTER_LINEAR,mask_interpolation=cv2.INTER_NEAREST),
        A.RandomRotate90(),A.HorizontalFlip(),A.VerticalFlip(0.3),
        A.ColorJitter(0.1,0.1,0.1,0.05,0.4),A.Normalize(),ToTensorV2()
    ])
    val_tf   = A.Compose([A.Resize(Config.IMAGE_HEIGHT,Config.IMAGE_WIDTH),
                          A.Normalize(),ToTensorV2()])

    # Datasets & loaders
    tr_ds = CloudPatchDatasetBalanced(Config.TRAIN_IMG_DIR,Config.TRAIN_MASK_DIR,
                                      patch_size=128,stride=128,min_fg_ratio=0.1,transform=train_tf)
    tpp = np.full(6, 1e12)   # target_pixels_per_class
    tpp[4] *= 5              # ← clase 4: 5× más presupuesto
    tr_samp = PixelBalancedSampler(tr_ds, tpp,
                                Config.BATCH_SIZE, drop_last=False)
    tr_ld = DataLoader(tr_ds, batch_sampler=tr_samp, num_workers=Config.NUM_WORKERS,
                       pin_memory=Config.PIN_MEMORY)

    val_ds = CloudPatchDatasetBalanced(Config.VAL_IMG_DIR,Config.VAL_MASK_DIR,
                                       patch_size=128,stride=128,transform=val_tf)
    val_ld = DataLoader(val_ds,batch_size=Config.BATCH_SIZE,shuffle=False,
                        num_workers=Config.NUM_WORKERS,pin_memory=Config.PIN_MEMORY)

    imprimir_distribucion_clases_post_augmentation(tr_ld,6,
        "Distribución de clases en ENTRENAMIENTO (post-aug)")

    # Modelo + pérdida
    model = CloudDeepLabV3Plus(num_classes=6).to(Config.DEVICE)
    w = torch.tensor(compute_class_weights(tr_ds), device=Config.DEVICE)
    BOOST = 3.0                     # prueba con 2-4; 0 = sin cambio
    w[4] = min(w[4] * BOOST, Config.MAX_W_CLS)

    loss_fn = AsymFocalTverskyLoss(class_weights=w,
                                eps=Config.EPS_TVERSKY).to(Config.DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS)
    scaler    = GradScaler(enabled=Config.USE_AMP)

    best_miou = -1
    for epoch in range(Config.NUM_EPOCHS):
        print(f"\n╭─ Epoch {epoch+1}/{Config.NUM_EPOCHS} ──────────────────────────")
        train_one_epoch(tr_ld, model, loss_fn, optimizer, scaler)
        miou, dice = check_metrics(val_ld, model, device=Config.DEVICE)
        scheduler.step()
        if miou>best_miou:
            best_miou=miou
            torch.save({"epoch":epoch,"state_dict":model.state_dict(),
                        "best_mIoU":best_miou}, Config.MODEL_SAVE_PATH)
            print(f"💾  Nuevo mejor mIoU: {miou:.4f}  (modelo guardado)")

    # ─ Fin de entrenamiento → cargar mejor modelo y evaluar ─
    if os.path.exists(Config.MODEL_SAVE_PATH):
        ckpt=torch.load(Config.MODEL_SAVE_PATH,map_location=Config.DEVICE)
        model.load_state_dict(ckpt["state_dict"])
        print(f"\n🔄  Modelo recargado del epoch {ckpt['epoch']}  (mIoU={ckpt['best_mIoU']:.4f})")
    print("\n📊  Evaluación final sobre VALIDACIÓN")
    miou,dice = check_metrics(val_ld, model, device=Config.DEVICE)
    print(f"Resultado final  →  mIoU={miou:.4f} | Dice={dice:.4f}")

if __name__ == "__main__":
    main()