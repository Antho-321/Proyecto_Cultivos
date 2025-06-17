import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# =================================================================================
# 1. MÓDULO DE ATENCIÓN (### MODIFICADO ###)
# =================================================================================
class ChannelAttention(nn.Module):
    """Módulo de Atención de Canal para enfocar qué características son importantes."""
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    """
    Módulo de Atención Espacial para enfocar dónde están las características importantes.
    (### MODIFICADO ###) Kernel reducido para un enfoque más localizado en áreas pequeñas.
    """
    def __init__(self, kernel_size=3): # ### MODIFICADO ### Kernel size de 7 a 3
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 5, 7), 'Kernel size must be 3, 5, or 7'
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class AttentionModule(nn.Module):
    """Módulo de Atención completo que combina Canal y Espacial."""
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=3):
        super(AttentionModule, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x) # Aplica atención de canal
        x = x * self.spatial_attention(x) # Aplica atención espacial
        return x

# =================================================================================
# 2. MÓDULO ASPP (Sin cambios en la clase, pero se cambiarán los `rates` al instanciar)
# =================================================================================
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        super(ASPPConv, self).__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ))
        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))
        modules.append(ASPPPooling(in_channels, out_channels))
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

# =================================================================================
# 3. BLOQUE DE DECODIFICADOR FPN (### MODIFICADO ###)
# =================================================================================
class DecoderBlock(nn.Module):
    """
    Bloque decodificador mejorado.
    (### MODIFICADO ###) Usa ConvTranspose2d para un upsampling aprendible y más nítido.
    """
    def __init__(self, in_channels_skip, in_channels_up, out_channels, use_attention=True):
        super(DecoderBlock, self).__init__()

        # --- NUEVO: Capa de upsampling aprendible ---
        # Reemplaza F.interpolate con una convolución transpuesta para aprender
        # a reconstruir detalles finos de manera más efectiva.
        self.upsample = nn.ConvTranspose2d(
            in_channels_up, in_channels_up, kernel_size=2, stride=2
        )
        
        total_in_channels = in_channels_skip + in_channels_up
        
        self.use_attention = use_attention
        if use_attention:
            # ### RECOMENDACIÓN ### kernel_size=3 para enfoque en detalles pequeños
            self.attention = AttentionModule(in_channels=in_channels_skip, kernel_size=3)

        # Bloque convolucional para refinar las características fusionadas
        self.conv_fuse = nn.Sequential(
            nn.Conv2d(total_in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x_skip, x_up):
        # x_up es la característica de la capa anterior, x_skip es del backbone
        x_up = self.upsample(x_up)
        
        if self.use_attention:
            x_skip = self.attention(x_skip)
        
        x_concat = torch.cat([x_up, x_skip], dim=1)
        return self.conv_fuse(x_concat)


# =================================================================================
# 4. ARQUITECTURA PRINCIPAL: CloudDeepLabV3+ (### MODIFICADO ###)
# =================================================================================
class CloudDeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=1):
        """
        (### MODIFICADO ###) 
        1. Se han ajustado las tasas de dilatación de ASPP para capturar mejor los detalles pequeños.
        2. Se ha añadido un cabezal de segmentación más robusto para refinar la salida final.
        """
        # ### MODIFICADO ### Tasas de dilatación más pequeñas para detalles finos.
        atrous_rates=(6, 12, 18)
        
        super(CloudDeepLabV3Plus, self).__init__()
        
        # --- Backbone: EfficientNetV2-S ---
        self.backbone = timm.create_model(
            'tf_efficientnetv2_s.in21k',
            pretrained=True,
            features_only=True,
            out_indices=(0, 1, 2, 3) # Stride 2, 4, 8, 16
        )
        
        backbone_channels = self.backbone.feature_info.channels()
        # [24, 48, 64, 160]
        
        # --- Módulo ASPP para el nivel más profundo ---
        self.aspp = ASPP(in_channels=backbone_channels[3], atrous_rates=atrous_rates, out_channels=256)
        
        # --- Decoder FPN-Style ---
        decoder_out_channels = [128, 64, 48]
        
        self.decoder_block3 = DecoderBlock(
            in_channels_skip=backbone_channels[2], # Stride 8
            in_channels_up=256, # Salida del ASPP
            out_channels=decoder_out_channels[0]
        )
        self.decoder_block2 = DecoderBlock(
            in_channels_skip=backbone_channels[1], # Stride 4
            in_channels_up=decoder_out_channels[0],
            out_channels=decoder_out_channels[1]
        )
        self.decoder_block1 = DecoderBlock(
            in_channels_skip=backbone_channels[0], # Stride 2
            in_channels_up=decoder_out_channels[1],
            out_channels=decoder_out_channels[2]
        )
        
        # --- (### MODIFICADO ###) Cabezal de Segmentación Mejorado ---
        # En lugar de una sola conv 1x1, usamos un bloque convolucional para
        # un refinamiento espacial final antes de la clasificación de píxeles.
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(decoder_out_channels[2], 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )
        
        # --- Upsampling Final ---
        self.final_upsample = nn.UpsamplingBilinear2d(scale_factor=2)


    def forward(self, x):
        # 1. Obtener características del backbone en múltiples escalas
        features = self.backbone(x)

        # 2. Procesar características de alto nivel con ASPP
        aspp_output = self.aspp(features[3])
        
        # 3. Decodificar fusionando progresivamente
        decoder_out3 = self.decoder_block3(x_skip=features[2], x_up=aspp_output)
        decoder_out2 = self.decoder_block2(x_skip=features[1], x_up=decoder_out3)
        decoder_out1 = self.decoder_block1(x_skip=features[0], x_up=decoder_out2)
        
        # 4. (### MODIFICADO ###) Generar el mapa de segmentación con el cabezal mejorado
        logits = self.segmentation_head(decoder_out1)
        
        # 5. Sobredimensionar a tamaño de entrada original
        final_logits_upsampled = self.final_upsample(logits)
        
        return final_logits_upsampled