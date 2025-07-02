import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# =================================================================================
# 1. MÓDULO DE ATENCIÓN MEJORADO
# =================================================================================
class ChannelAttention(nn.Module):
    """Atención de canal."""
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.fc(self.avg_pool(x))
        mx  = self.fc(self.max_pool(x))
        return self.sigmoid(avg + mx)

class EnhancedSpatialAttention(nn.Module):
    """Atención espacial multi-escala."""
    def __init__(self):
        super().__init__()
        # convoluciones multi-escala
        self.conv1 = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(2, 1, kernel_size=5, padding=2, bias=False)
        self.conv3 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.final_conv = nn.Conv2d(3, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        inp = torch.cat([avg, mx], dim=1)
        x1 = self.conv1(inp)
        x2 = self.conv2(inp)
        x3 = self.conv3(inp)
        cat = torch.cat([x1, x2, x3], dim=1)
        return self.sigmoid(self.final_conv(cat))

class AttentionModule(nn.Module):
    """Combina atención de canal y espacial mejorada."""
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.channel = ChannelAttention(in_channels, reduction_ratio)
        self.spatial = EnhancedSpatialAttention()

    def forward(self, x):
        x = x * self.channel(x)
        x = x * self.spatial(x)
        return x

# =================================================================================
# 2. ASPP (con tasas de dilatación más pequeñas)
# =================================================================================
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self, x):
        size = x.shape[-2:]
        return F.interpolate(super().forward(x), size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super().__init__()
        modules = [ nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )]
        for rate in atrous_rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))
        modules.append(ASPPPooling(in_channels, out_channels))
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(len(modules) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        res = [conv(x) for conv in self.convs]
        return self.project(torch.cat(res, dim=1))

# =================================================================================
# 3. BLOQUE DECODIFICADOR FPN
# =================================================================================
class DecoderBlock(nn.Module):
    def __init__(self, in_channels_skip, in_channels_up, out_channels, use_attention=True):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels_up, in_channels_up, 2, stride=2)
        self.align = (nn.Conv2d(in_channels_skip, out_channels, 1, bias=False)
                      if in_channels_skip != out_channels else nn.Identity())
        total_in = in_channels_skip + in_channels_up
        self.att = AttentionModule(in_channels_skip) if use_attention else None
        self.fuse = nn.Sequential(
            nn.Conv2d(total_in, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.relu = nn.ReLU()

    def forward(self, x_skip, x_up):
        x = self.upsample(x_up)  
        skip = self.att(x_skip)  
        skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=False)  
        cat = torch.cat([x, skip], dim=1)
        fused = self.fuse(cat)
        res = self.align(x_skip)
        res = F.interpolate(res, size=fused.shape[-2:], mode='bilinear', align_corners=False)
        return self.relu(fused + res)

# =================================================================================
# 4. ARQUITECTURA PRINCIPAL: CloudDeepLabV3+ MEJORADO
# =================================================================================
class CloudDeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        # shallow feature for small objects
        self.shallow_conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        # backbone con salida multi-escala y menor downsampling
        self.backbone = timm.create_model(
            'tf_efficientnetv2_s.in21k',
            pretrained=True,
            features_only=True,
            out_indices=(0, 1, 2, 3, 4),
            output_stride=8
        )
        chans = self.backbone.feature_info.channels()  # [24, 48, 64, 160, 256]

        # ASPP con tasas más finas
        # deep = feats[-1] tiene chans[-1] canales, no chans[3]
        self.aspp = ASPP(in_channels=chans[-1], atrous_rates=(1,2,4,8), out_channels=256)

        # bloques de decodificador
        dec_ch = [128, 64, 48]
        self.decoder_block3 = DecoderBlock(chans[3], 256, dec_ch[0])
        self.decoder_block2 = DecoderBlock(chans[2], dec_ch[0], dec_ch[1])
        self.decoder_block1 = DecoderBlock(chans[1], dec_ch[1], dec_ch[2])

        # cabezal de segmentación principal
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(dec_ch[2], 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, num_classes, 1)
        )

        # cabezales auxiliares
        self.aux_head_3 = nn.Conv2d(dec_ch[0], num_classes, 1)
        self.aux_head_2 = nn.Conv2d(dec_ch[1], num_classes, 1)
        self.final_upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        # detección de objetos pequeños
        in_small = chans[0] + 16
        self.small_object_head = nn.Sequential(
            nn.Conv2d(in_small, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, num_classes, 1)
        )

    def forward(self, x):
        # pequeña rama de alta resolución
        shallow = self.shallow_conv(x)

        feats = self.backbone(x)
        # feats = [stride2, stride4, stride8, stride8, stride8]
        # choose the *last* one for ASPP:
        deep = feats[-1]                # H/8 × W/8
        aspp_out = self.aspp(deep)      # H/8 × W/8
        # Now decoder3 should fuse with feats[-2], which is also H/8×W/8:
        d3 = self.decoder_block3(feats[-2], aspp_out)
        # That up-samples to H/4×W/4 and then fuses with feats[-3], which *must* be H/4×W/4, etc.
        d2 = self.decoder_block2(feats[-3], d3)
        d1 = self.decoder_block1(feats[-4], d2)

        # cabezal de pequeños objetos
        skip0   = feats[0]                              # (B, C0, H/2, W/2)
        # Redimensionamos shallow a H/2×W/2 y lo juntamos
        shallow_ds = F.interpolate(shallow, skip0.shape[-2:], mode='bilinear', align_corners=False)
        merged0   = torch.cat([skip0, shallow_ds], dim=1)  # (B, C0+16, H/2, W/2)
        small_logits = self.small_object_head(merged0)
        small_logits = F.interpolate(small_logits, size=x.shape[-2:], mode='bilinear', align_corners=False)

        # segmentación principal
        seg_logits = self.segmentation_head(d1)
        main_up = seg_logits

        # combinación ponderada
        combined = main_up + 0.3 * small_logits

        if self.training:
            aux3 = F.interpolate(self.aux_head_3(d3), size=x.shape[-2:], mode='bilinear', align_corners=False)
            aux2 = F.interpolate(self.aux_head_2(d2), size=x.shape[-2:], mode='bilinear', align_corners=False)
            return combined, aux3, aux2, small_logits

        return combined