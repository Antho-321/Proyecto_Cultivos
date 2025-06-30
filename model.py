import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# =================================================================================
# 1. MÓDULO DE ATENCIÓN (con normalización de pesos opcional y uso de Softmax)
# =================================================================================
class ChannelAttention(nn.Module):
    """Módulo de Atención de Canal para enfocar qué características son importantes."""
    def __init__(self, in_channels, reduction_ratio=16, use_softmax=True):
        super(ChannelAttention, self).__init__()
        self.use_softmax = use_softmax
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
        out = avg_out + max_out           # shape (B, C, 1, 1)

        if self.use_softmax:
            out = F.softmax(out, dim=1)   # attention weights sum to 1
        else:
            out = torch.sigmoid(out)      # independent gating
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7, use_channel_weight=True):  
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        
        # Parámetros de peso aprendibles
        self.channel_weight = nn.Parameter(torch.ones(2)) if use_channel_weight else None

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        if self.channel_weight is not None:
            weighted_avg = avg_out * self.channel_weight[0]
            weighted_max = max_out * self.channel_weight[1]
        else:
            weighted_avg = avg_out
            weighted_max = max_out
        
        x = torch.cat([weighted_avg, weighted_max], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class AttentionModule(nn.Module):
    """Módulo de Atención completo que combina Canal y Espacial."""
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7, use_softmax=False):
        super(AttentionModule, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio, use_softmax)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)  # Aplica atención de canal
        x = x * self.spatial_attention(x)  # Aplica atención espacial
        return x


# =================================================================================
# 2. ASPP POOLING (del original, reutilizado)
# =================================================================================
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


# =================================================================================
# 3. SEPARABLE ASPP CONV
# =================================================================================
class SeparableASPPConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, 3, 
                                   padding=dilation, dilation=dilation, 
                                   groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)


# =================================================================================
# 4. SQUEEZE-AND-EXCITATION BLOCK
# =================================================================================
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# =================================================================================
# 5. ENHANCED ASPP (usando SeparableASPPConv + SEBlock)
# =================================================================================
class EnhancedASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256, dropout_rate=0.2):
        super().__init__()
        modules = []
        
        # 1x1 conv
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ))
        
        # Separable atrous convolutions
        for rate in atrous_rates:
            modules.append(SeparableASPPConv(in_channels, out_channels, rate))
            
        # Global average pooling
        modules.append(ASPPPooling(in_channels, out_channels))
        
        self.convs = nn.ModuleList(modules)
        
        # Enhanced projection with squeeze-excitation
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            SEBlock(out_channels),  # Add SE attention
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        res = [conv(x) for conv in self.convs]
        res = torch.cat(res, dim=1)
        return self.project(res)


# =================================================================================
# 6. BOUNDARY-AWARE DECODER BLOCK
# =================================================================================
class BoundaryAwareDecoderBlock(nn.Module):
    def __init__(self, in_channels_skip, in_channels_up, out_channels, dropout_rate=0.2):
        super().__init__()
        
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels_up, in_channels_up, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(in_channels_up),
            nn.ReLU()
        )
        
        # Boundary detection branch
        self.boundary_conv = nn.Sequential(
            nn.Conv2d(in_channels_skip, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )
        
        # Main fusion path
        total_in = in_channels_skip + in_channels_up
        self.fusion = nn.Sequential(
            nn.Conv2d(total_in, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            SEBlock(out_channels),  # Add SE attention
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Residual alignment
        self.align = nn.Conv2d(in_channels_skip, out_channels, 1, bias=False) \
                     if in_channels_skip != out_channels else nn.Identity()
            
    def forward(self, x_skip, x_up):
        x_up = self.upsample(x_up)
        
        # Detect boundaries for attention
        boundary_mask = self.boundary_conv(x_skip)
        x_skip_enhanced = x_skip * (1 + boundary_mask)
        
        # Concatenate and fuse
        x_concat = torch.cat([x_up, x_skip_enhanced], dim=1)
        x_fused = self.fusion(x_concat)
        
        # Residual connection
        x_residual = self.align(x_skip)
        if x_residual.shape[-2:] != x_fused.shape[-2:]:
            x_residual = F.interpolate(x_residual,
                                       size=x_fused.shape[-2:],
                                       mode='bilinear',
                                       align_corners=False)
        
        return x_fused + x_residual


# =================================================================================
# 7. ARQUITECTURA PRINCIPAL: CloudDeepLabV3+ (usando EnhancedASPP y BoundaryAwareDecoderBlock)
# =================================================================================
class CloudDeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=1, dropout_rate=0.2):
        super().__init__()
        atrous_rates = (2, 6, 12)
        
        # --- Backbone: EfficientNetV2-S ---
        self.backbone = timm.create_model(
            'tf_efficientnetv2_s.in21k',
            pretrained=True,
            features_only=True,
            out_indices=(0, 1, 2, 3)
        )
        backbone_channels = self.backbone.feature_info.channels()
        
        # --- Enhanced ASPP ---
        self.aspp = EnhancedASPP(
            in_channels=backbone_channels[3],
            atrous_rates=atrous_rates,
            out_channels=256,
            dropout_rate=dropout_rate
        )
        
        # --- Boundary-aware decoder blocks ---
        decoder_channels = [128, 64, 48]
        self.decoder_block3 = BoundaryAwareDecoderBlock(
            in_channels_skip=backbone_channels[2],
            in_channels_up=256,
            out_channels=decoder_channels[0],
            dropout_rate=dropout_rate
        )
        self.decoder_block2 = BoundaryAwareDecoderBlock(
            in_channels_skip=backbone_channels[1],
            in_channels_up=decoder_channels[0],
            out_channels=decoder_channels[1],
            dropout_rate=dropout_rate
        )
        self.decoder_block1 = BoundaryAwareDecoderBlock(
            in_channels_skip=backbone_channels[0],
            in_channels_up=decoder_channels[1],
            out_channels=decoder_channels[2],
            dropout_rate=dropout_rate
        )
        
        # --- Segmentation head and auxiliary heads ---
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(decoder_channels[2], 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, num_classes, 1)
        )
        self.aux_head_3 = nn.Conv2d(decoder_channels[0], num_classes, 1)
        self.aux_head_2 = nn.Conv2d(decoder_channels[1], num_classes, 1)
        self.final_upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.training = True

    def forward(self, x):
        features = self.backbone(x)
        aspp_output = self.aspp(features[3])
        
        d3 = self.decoder_block3(features[2], aspp_output)
        d2 = self.decoder_block2(features[1], d3)
        d1 = self.decoder_block1(features[0], d2)
        
        logits = self.segmentation_head(d1)
        out = self.final_upsample(logits)
        
        if self.training:
            aux3 = F.interpolate(self.aux_head_3(d3),
                                 size=x.shape[-2:], mode='bilinear', align_corners=False)
            aux2 = F.interpolate(self.aux_head_2(d2),
                                 size=x.shape[-2:], mode='bilinear', align_corners=False)
            return out, aux3, aux2
        
        return out