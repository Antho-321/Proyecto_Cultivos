# model.py (PyTorch Version)

import torch
import torch.nn as nn
import timm

class ConvBlock(nn.Module):
    """
    Standard 3x3 convolution block with ReLU and Batch Norm.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UpBlock(nn.Module):
    """
    Upsampling block (ConvTranspose2d) followed by a ConvBlock.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # x2 is the skip connection from the encoder
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class EfficientNetUnet(nn.Module):
    def __init__(self, num_classes=6, pretrained=True):
        super().__init__()

        # --- ENCODER ---
        # Use timm to get an EfficientNetB0 backbone with feature extraction capabilities
        self.encoder = timm.create_model(
            'efficientnet_b0',
            pretrained=pretrained,
            features_only=True, # This is key! It returns feature maps from intermediate layers
        )

        # Get the output channel sizes from the encoder, which we'll need for the decoder
        encoder_channels = self.encoder.feature_info.channels() # e.g., [16, 24, 40, 112, 320]

        # --- DECODER ---
        # The decoder will upsample from the final encoder feature map, concatenating
        # with skip connections from earlier layers.
        # `encoder_channels` are ordered from earliest to latest layer.
        self.up1 = UpBlock(encoder_channels[4], encoder_channels[3]) # From 320 -> 112 channels
        self.up2 = UpBlock(encoder_channels[3], encoder_channels[2]) # From 112 -> 40 channels
        self.up3 = UpBlock(encoder_channels[2], encoder_channels[1]) # From 40  -> 24 channels
        self.up4 = UpBlock(encoder_channels[1], encoder_channels[0]) # From 24  -> 16 channels

        # A final upsampling block to get to a higher resolution before the output layer
        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(encoder_channels[0], 16, kernel_size=2, stride=2),
            ConvBlock(16, 16)
        )

        # --- OUTPUT LAYER ---
        # Final convolution to get the desired number of class logits
        self.final_conv = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        # --- ENCODER FORWARD PASS ---
        # The `features_only=True` encoder returns a list of tensors
        skip_connections = self.encoder(x)
        # The last feature map is the input to the decoder's first block
        encoder_output = skip_connections[-1]

        # --- DECODER FORWARD PASS ---
        # We work backwards through the skip connections list
        d1 = self.up1(encoder_output, skip_connections[3])
        d2 = self.up2(d1, skip_connections[2])
        d3 = self.up3(d2, skip_connections[1])
        d4 = self.up4(d3, skip_connections[0])
        d5 = self.up5(d4) # No skip connection here

        # --- FINAL OUTPUT ---
        logits = self.final_conv(d5)
        return logits