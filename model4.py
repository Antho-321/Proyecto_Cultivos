import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# =================================================================================
# 1. MÓDULO DE ATENCIÓN (sin cambios)
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
    def __init__(self, kernel_size=7):  # Revert to 7 for better context
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        
        # Add learnable weighting
        self.channel_weight = nn.Parameter(torch.ones(2))

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Weighted combination
        weighted_avg = avg_out * self.channel_weight[0]
        weighted_max = max_out * self.channel_weight[1]
        
        x = torch.cat([weighted_avg, weighted_max], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class AttentionModule(nn.Module):
    """Módulo de Atención completo que combina Canal y Espacial."""
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=3):
        super(AttentionModule, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)  # Aplica atención de canal
        x = x * self.spatial_attention(x)  # Aplica atención espacial
        return x

# =================================================================================
# 2. MÓDULO ASPP (Añadido Dropout)
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
        modules = [
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
        ]
        for rate in atrous_rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))
        modules.append(ASPPPooling(in_channels, out_channels))
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.3)  # Añadido Dropout con probabilidad 0.3
        )

    def forward(self, x):
        res = [conv(x) for conv in self.convs]
        res = torch.cat(res, dim=1)
        return self.project(res)

# =================================================================================
# 3. BLOQUE DE DECODIFICADOR FPN (Añadido Dropout)
# =================================================================================
class DecoderBlock(nn.Module):
    def __init__(self, in_channels_skip, in_channels_up, out_channels, use_attention=True):
        super(DecoderBlock, self).__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels_up, in_channels_up, kernel_size=2, stride=2
        )
        self.align_channels = (
            nn.Conv2d(in_channels_skip, out_channels, 1, bias=False)
            if in_channels_skip != out_channels
            else nn.Identity()
        )
        total_in_channels = in_channels_skip + in_channels_up

        self.use_attention = use_attention
        if use_attention:
            self.attention = AttentionModule(in_channels=in_channels_skip, kernel_size=7)

        self.conv_fuse = nn.Sequential(
            nn.Conv2d(total_in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.final_relu = nn.ReLU()

    def forward(self, x_skip, x_up):
        x_up = self.upsample(x_up)
        x_skip_att = self.attention(x_skip) if self.use_attention else x_skip
        x_concat = torch.cat([x_up, x_skip_att], dim=1)
        x_fused = self.conv_fuse(x_concat)
        x_residual = self.align_channels(x_skip)
        x_residual = F.interpolate(x_residual, size=x_fused.shape[-2:], mode='bilinear', align_corners=False)
        return self.final_relu(x_fused + x_residual)

# =================================================================================
# 4. ARQUITECTURA PRINCIPAL: CloudDeepLabV3+ (con Dropout añadido y _init_weights)
# =================================================================================
class CloudDeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=1):
        super(CloudDeepLabV3Plus, self).__init__()
        atrous_rates = (2, 6, 12)

        # Backbone: EfficientNetV2-S inicializado aleatoriamente
        self.backbone = timm.create_model(
            'tf_efficientnetv2_s.in21k',
            pretrained=False,
            features_only=True,
            out_indices=(0, 1, 2, 3)  # Stride 2, 4, 8, 16
        )
        backbone_channels = self.backbone.feature_info.channels()
        # e.g. [24, 48, 64, 160]

        # Módulo ASPP
        self.aspp = ASPP(in_channels=backbone_channels[3], atrous_rates=atrous_rates, out_channels=256)

        # Decoder FPN-Style
        decoder_out_channels = [128, 64, 48]
        self.decoder_block3 = DecoderBlock(backbone_channels[2], 256, decoder_out_channels[0])
        self.decoder_block2 = DecoderBlock(backbone_channels[1], decoder_out_channels[0], decoder_out_channels[1])
        self.decoder_block1 = DecoderBlock(backbone_channels[0], decoder_out_channels[1], decoder_out_channels[2])

        # Cabezal de segmentación
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(decoder_out_channels[2], 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )

        # Cabezas auxiliares para supervisión
        self.aux_head_3 = nn.Conv2d(decoder_out_channels[0], num_classes, 1)
        self.aux_head_2 = nn.Conv2d(decoder_out_channels[1], num_classes, 1)

        # Upsampling final
        self.final_upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        # Inicialización de pesos personalizada
        self.apply(self._init_weights)

    def forward(self, x):
        features = self.backbone(x)

        aspp_output = self.aspp(features[3])
        d3 = self.decoder_block3(features[2], aspp_output)
        d2 = self.decoder_block2(features[1], d3)
        d1 = self.decoder_block1(features[0], d2)

        logits     = self.segmentation_head(d1)
        final_map  = self.final_upsample(logits)

        if self.training:
            aux3 = F.interpolate(self.aux_head_3(d3), size=x.shape[-2:], mode='bilinear', align_corners=False)
            aux2 = F.interpolate(self.aux_head_2(d2), size=x.shape[-2:], mode='bilinear', align_corners=False)
            return final_map, aux3, aux2

        return final_map

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.zeros_(m.bias)