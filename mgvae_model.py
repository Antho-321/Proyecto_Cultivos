# mgvae_model.py

import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, input_channels=3, latent_dim=128):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        # --- Codificador ---
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1), # -> 128x128
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # -> 64x64
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # -> 32x32
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),# -> 16x16
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Capas para predecir la media (mu) y la varianza logarítmica (log_var)
        self.fc_mu = nn.Linear(256 * 16 * 16, latent_dim)
        self.fc_log_var = nn.Linear(256 * 16 * 16, latent_dim)

        # --- Decodificador ---
        self.decoder_input = nn.Linear(latent_dim, 256 * 16 * 16)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # -> 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # -> 64x64
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # -> 128x128
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1), # -> 256x256
            nn.Sigmoid()  # Salida en el rango [0, 1]
        )

    def reparameterize(self, mu, log_var):
        """Truco de reparametrización para permitir el backpropagation."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Codificar
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        
        # Reparametrizar para obtener el vector latente z
        z = self.reparameterize(mu, log_var)
        
        # Decodificar
        z_projected = self.decoder_input(z).view(-1, 256, 16, 16)
        reconstruction = self.decoder(z_projected)
        
        return reconstruction, mu, log_var

# Función de pérdida para el VAE
def vae_loss(recon_x, x, mu, log_var):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD