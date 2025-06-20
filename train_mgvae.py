# train_mgvae.py

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image
from tqdm import tqdm
import os
import numpy as np
from PIL import Image

# Importaciones de tu proyecto
from train import CloudDataset # Reutilizamos tu dataset
from mgvae_model import VAE, vae_loss
from config import Config

# --- CONFIGURACIÓN PARA MGVAE ---
MAJORITY_CLASS_ID = 0  # Ejemplo: ID de la clase mayoritaria (cielo)
MINORITY_CLASS_ID = 4  # Ejemplo: ID de la clase minoritaria a aumentar
NUM_SAMPLES_TO_GENERATE = 500 # Cuántas imágenes nuevas crear
MGVAE_EPOCHS_MAJORITY = 25   # Épocas para pre-entrenar con la mayoría
MGVAE_EPOCHS_MINORITY = 40   # Épocas para ajustar con la minoría
MGVAE_BATCH_SIZE = 16
DEVICE = Config.DEVICE

# Directorios para los datos generados
SAVE_DIR_SYNTHETIC_IMG = "data/synthetic_images"
SAVE_DIR_SYNTHETIC_MASK = "data/synthetic_masks"
os.makedirs(SAVE_DIR_SYNTHETIC_IMG, exist_ok=True)
os.makedirs(SAVE_DIR_SYNTHETIC_MASK, exist_ok=True)


def get_class_specific_indices(dataset, class_id, min_pixel_percentage=0.1):
    """
    Recorre el dataset y devuelve los índices de las imágenes que contienen
    suficientes píxeles de la clase especificada.
    """
    print(f"Buscando imágenes con al menos {min_pixel_percentage*100}% de píxeles de la clase {class_id}...")
    indices = []
    for i in tqdm(range(len(dataset))):
        _, mask = dataset[i] # El dataset devuelve (imagen, mascara)
        total_pixels = mask.numel()
        class_pixels = torch.sum(mask == class_id)
        if (class_pixels / total_pixels) > min_pixel_percentage:
            indices.append(i)
    print(f"Encontradas {len(indices)} imágenes para la clase {class_id}.")
    return indices


def train_vae_phase(model, loader, optimizer, epochs, phase_name):
    """Función genérica para entrenar el VAE."""
    model.train()
    print(f"\n--- Iniciando Fase de Entrenamiento: {phase_name} ---")
    for epoch in range(epochs):
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        total_loss = 0.0
        for i, (data, _) in enumerate(loop): # No necesitamos la máscara para entrenar el VAE de imágenes
            data = data.to(DEVICE)
            optimizer.zero_grad()
            recon, mu, log_var = model(data)
            loss = vae_loss(recon, data, mu, log_var)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item() / len(data))
        print(f"Epoch {epoch+1} Loss: {total_loss / len(loader.dataset)}")


def main():
    # 1. Cargar el dataset completo para poder filtrar por clase
    full_train_dataset = CloudDataset(
        image_dir=Config.TRAIN_IMG_DIR,
        mask_dir=Config.TRAIN_MASK_DIR
    )

    # 2. Obtener los datos para la clase MAYORITARIA
    majority_indices = get_class_specific_indices(full_train_dataset, MAJORITY_CLASS_ID)
    majority_subset = Subset(full_train_dataset, majority_indices)
    majority_loader = DataLoader(majority_subset, batch_size=MGVAE_BATCH_SIZE, shuffle=True)

    # 3. FASE 1: Pre-entrenar el VAE con la clase MAYORITARIA
    vae_model = VAE(input_channels=3, latent_dim=128).to(DEVICE)
    optimizer = optim.Adam(vae_model.parameters(), lr=1e-4)
    train_vae_phase(vae_model, majority_loader, optimizer, MGVAE_EPOCHS_MAJORITY, "Mayoría")

    # 4. Obtener los datos para la clase MINORITARIA
    minority_indices = get_class_specific_indices(full_train_dataset, MINORITY_CLASS_ID)
    minority_subset = Subset(full_train_dataset, minority_indices)
    minority_loader = DataLoader(minority_subset, batch_size=MGVAE_BATCH_SIZE, shuffle=True)
    
    # 5. FASE 2: Ajustar (fine-tune) el VAE con la clase MINORITARIA
    # El modelo ya está pre-entrenado, continuamos desde ahí
    train_vae_phase(vae_model, minority_loader, optimizer, MGVAE_EPOCHS_MINORITY, "Minoría (Ajuste Fino)")

    # 6. GENERACIÓN: Usar el modelo final para generar nuevas muestras
    print(f"\n--- Generando {NUM_SAMPLES_TO_GENERATE} muestras sintéticas para la clase {MINORITY_CLASS_ID} ---")
    vae_model.eval()
    with torch.no_grad():
        for i in tqdm(range(NUM_SAMPLES_TO_GENERATE)):
            # Muestreamos del espacio latente (distribución normal estándar)
            z = torch.randn(1, vae_model.latent_dim).to(DEVICE)
            # Generamos la imagen
            synthetic_img = vae_model.decoder(vae_model.decoder_input(z).view(-1, 256, 16, 16)).cpu()
            
            # Guardamos la imagen sintética
            save_image(synthetic_img, os.path.join(SAVE_DIR_SYNTHETIC_IMG, f"synthetic_{i}.png"))
            
            # Creamos y guardamos una máscara simple correspondiente
            # Aquí asumimos que la imagen generada es 100% de la clase minoritaria
            mask_np = np.full((256, 256), fill_value=MINORITY_CLASS_ID, dtype=np.uint8)
            mask_pil = Image.fromarray(mask_np)
            mask_pil.save(os.path.join(SAVE_DIR_SYNTHETIC_MASK, f"synthetic_{i}_mask.png"))
            
    print("¡Generación completada!")
    print(f"Imágenes guardadas en: {SAVE_DIR_SYNTHETIC_IMG}")
    print(f"Máscaras guardadas en: {SAVE_DIR_SYNTHETIC_MASK}")


if __name__ == "__main__":
    main()