import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# =================================================================================
# 1. MÓDULO DE ATENCIÓN (Sin cambios)
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
    """Módulo de Atención Espacial para enfocar dónde están las características importantes."""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class AttentionModule(nn.Module):
    """Módulo de Atención completo que combina Canal y Espacial."""
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(AttentionModule, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x) # Aplica atención de canal
        x = x * self.spatial_attention(x) # Aplica atención espacial
        return x

# =================================================================================
# 2. MÓDULO ASPP (Sin cambios)
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
# 3. ### NUEVO ### BLOQUE DE DECODIFICADOR FPN
# =================================================================================
class DecoderBlock(nn.Module):
    """Bloque para decodificar y fusionar características de forma progresiva."""
    def __init__(self, in_channels_skip, in_channels_up, out_channels, use_attention=True):
        super(DecoderBlock, self).__init__()
        # Canales de entrada = Canales del skip connection + Canales del upsampling
        self.conv_fuse = nn.Sequential(
            nn.Conv2d(in_channels_skip + in_channels_up, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.use_attention = use_attention
        if use_attention:
            self.attention = AttentionModule(in_channels=in_channels_skip)

    def forward(self, x_skip, x_up):
        # x_up es la característica de la capa anterior, x_skip es del backbone
        x_up = F.interpolate(x_up, size=x_skip.shape[-2:], mode='bilinear', align_corners=False)
        
        if self.use_attention:
            x_skip = self.attention(x_skip)
        
        x = torch.cat([x_up, x_skip], dim=1)
        return self.conv_fuse(x)

# =================================================================================
# 4. ARQUITECTURA PRINCIPAL: CloudDeepLabV3+ (### MODIFICADO ###)
# =================================================================================
class CloudDeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=1, atrous_rates=(12, 24, 36)):
        super(CloudDeepLabV3Plus, self).__init__()
        
        # --- Backbone: EfficientNetV2-S ---
        # (### MODIFICADO ###) Extraemos características de 4 niveles para una mejor fusión
        self.backbone = timm.create_model(
            'tf_efficientnetv2_s.in21k',
            pretrained=True,
            features_only=True,
            out_indices=(0, 1, 2, 3) # Stride 2, 4, 8, 16
        )
        
        backbone_channels = self.backbone.feature_info.channels()
        # [24, 48, 64, 160] para efficientnetv2_s con out_indices=(0, 1, 2, 3)
        
        # --- Módulo ASPP para el nivel más profundo ---
        self.aspp = ASPP(in_channels=backbone_channels[3], atrous_rates=atrous_rates, out_channels=256)
        
        # --- (### MODIFICADO ###) Decoder FPN-Style ---
        # Decodificadores para fusionar progresivamente las características del backbone
        # Usamos los canales del backbone y definimos los canales de salida de cada bloque
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
        
        # --- Clasificador Final ---
        # Upsampling final y clasificación
        self.final_conv = nn.Conv2d(decoder_out_channels[2], num_classes, 1)

    def forward(self, x):
        input_size = x.shape[-2:]
        
        # 1. Obtener características del backbone en múltiples escalas
        features = self.backbone(x)
        # features[0]: stride 2
        # features[1]: stride 4
        # features[2]: stride 8
        # features[3]: stride 16

        # 2. Procesar características de alto nivel con ASPP
        aspp_output = self.aspp(features[3])
        
        # 3. (### MODIFICADO ###) Decodificar fusionando progresivamente
        # Decodificar Stride 8
        decoder_out3 = self.decoder_block3(x_skip=features[2], x_up=aspp_output)
        # Decodificar Stride 4
        decoder_out2 = self.decoder_block2(x_skip=features[1], x_up=decoder_out3)
        # Decodificar Stride 2
        decoder_out1 = self.decoder_block1(x_skip=features[0], x_up=decoder_out2)
        
        # 4. Generar el mapa de segmentación final
        logits = self.final_conv(decoder_out1)
        
        # 5. Sobredimensionar a tamaño de entrada original
        final_logits_upsampled = F.interpolate(
            logits, 
            size=input_size, 
            mode='bilinear', 
            align_corners=False
        )
        
        return final_logits_upsampled