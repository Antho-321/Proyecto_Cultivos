import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# =================================================================================
# Helper: Bloque de convolución
# =================================================================================
def conv_block(in_channels, out_channels, kernel_size, dilation=1, bias=False):
    padding = ((kernel_size - 1) // 2) * dilation
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels,
                  kernel_size,
                  padding=padding,
                  dilation=dilation,
                  bias=bias),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

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
            nn.ReLU(inplace=True),
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
# 2. MÓDULO WASP (en lugar de ASPP)
# =================================================================================
class WASPModule(nn.Module):
    """
    Wide Atrous Spatial Pyramid (WASP) inspirado en tu definición.
      - dil: tupla de dilataciones
      - gp: incluir Global Pooling
      - ag: convolución adicional tras dilatación >1
      - att: aplicar SE al final
    """
    def __init__(self,
                 in_channels,
                 out_channels=256,
                 dil=(1, 2, 4, 8),
                 gp=True,
                 ag=True,
                 att=False):
        super().__init__()
        self.dil = dil
        self.gp = gp
        self.ag = ag
        self.att = att

        # Bloques sucesivos
        self.convs = nn.ModuleList()
        # Primer bloque 1×1
        self.convs.append(conv_block(in_channels, out_channels, 1))
        # Bloques dilatados y opcional "after-gate"
        for r in dil:
            # convolución dilatada
            self.convs.append(conv_block(out_channels, out_channels, 3, dilation=r))
            # convolución extra tras dilatación >1
            if ag and r > 1:
                self.convs.append(conv_block(out_channels, out_channels, 3))

        # Bloque de pooling global
        if gp:
            self.gp_conv = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        # Proyección tras concatenar
        branch_count = 1 + len(dil) + (1 if gp else 0)  # → 6
        self.project = conv_block(branch_count * out_channels, out_channels, 1)

        # SE (squeeze-and-excitation) opcional
        if att:
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(out_channels, out_channels // 16, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels // 16, out_channels, 1, bias=False),
                nn.Sigmoid()
            )

    def forward(self, x):
        branches = []

        # Primer branch
        y = self.convs[0](x)
        branches.append(y)
        idx = 1
        # Resto de branches
        for r in self.dil:
            b = self.convs[idx](branches[-1])
            idx += 1
            if self.ag and r > 1:
                b = self.convs[idx](b)
                idx += 1
            branches.append(b)

        # Global pooling branch
        if self.gp:
            g = self.gp_conv(x)
            g = F.interpolate(g, size=x.shape[-2:], mode='bilinear', align_corners=False)
            branches.append(g)

        # Concatenar y proyectar
        out = torch.cat(branches, dim=1)
        out = self.project(out)

        # SE
        if self.att:
            se = self.se(out)
            out = out * se

        return out

# =================================================================================
# 3. BLOQUE DECODIFICADOR FPN
# =================================================================================
class DecoderBlock(nn.Module):
    def __init__(self, in_channels_skip, in_channels_up, out_channels, use_attention=True):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels_up, in_channels_up, 2, stride=2)
        self.align = (
            nn.Conv2d(in_channels_skip, out_channels, 1, bias=False)
            if in_channels_skip != out_channels else nn.Identity()
        )
        self.att = AttentionModule(in_channels_skip) if use_attention else None
        total_in = in_channels_skip + in_channels_up
        self.fuse = nn.Sequential(
            nn.Conv2d(total_in, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_skip, x_up):
        x = self.upsample(x_up)
        skip = self.att(x_skip) if self.att else x_skip
        skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=False)
        cat = torch.cat([x, skip], dim=1)
        fused = self.fuse(cat)
        res = self.align(x_skip)
        res = F.interpolate(res, size=fused.shape[-2:], mode='bilinear', align_corners=False)
        return self.relu(fused + res)

# =================================================================================
# 4. ARQUITECTURA PRINCIPAL: CloudDeepLabV3+ MEJORADO CON WASP
# =================================================================================
class CloudDeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        # rama de resolución alta para objetos pequeños
        self.shallow_conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        # backbone multi-escala
        self.backbone = timm.create_model(
            'tf_efficientnetv2_s.in21k',
            pretrained=True,
            features_only=True,
            out_indices=(0, 1, 2, 3, 4),
            output_stride=8
        )
        chans = self.backbone.feature_info.channels()  # e.g. [24,48,64,160,256]

        # WASP en lugar de ASPP
        self.wasp = WASPModule(in_channels=chans[-1],
                               out_channels=256,
                               dil=(1,2,4,8),
                               gp=True,
                               ag=True,
                               att=False)

        # bloques de decodificador
        dec_ch = [128, 64, 48]
        self.decoder_block3 = DecoderBlock(chans[3], 256, dec_ch[0])
        self.decoder_block2 = DecoderBlock(chans[2], dec_ch[0], dec_ch[1])
        self.decoder_block1 = DecoderBlock(chans[1], dec_ch[1], dec_ch[2])

        # cabezal de segmentación principal
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(dec_ch[2], 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 1)
        )

        # cabezales auxiliares
        self.aux_head_3 = nn.Conv2d(dec_ch[0], num_classes, 1)
        self.aux_head_2 = nn.Conv2d(dec_ch[1], num_classes, 1)

        # detección de pequeños objetos
        in_small = chans[0] + 16
        self.small_object_head = nn.Sequential(
            nn.Conv2d(in_small, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 1)
        )

    def forward(self, x):
        shallow = self.shallow_conv(x)
        feats = self.backbone(x)
        deep = feats[-1]                  # H/8 × W/8
        wasp_out = self.wasp(deep)       # H/8 × W/8

        # FPN decoder
        d3 = self.decoder_block3(feats[-2], wasp_out)  # H/8→H/4
        d2 = self.decoder_block2(feats[-3], d3)        # H/4→H/2
        d1 = self.decoder_block1(feats[-4], d2)        # H/2→H

        # rama small objects
        skip0 = feats[0]                               # H/2 × W/2
        shallow_ds = F.interpolate(shallow,
                                   size=skip0.shape[-2:],
                                   mode='bilinear',
                                   align_corners=False)
        merged0 = torch.cat([skip0, shallow_ds], dim=1)
        small_logits = self.small_object_head(merged0)
        small_logits = F.interpolate(small_logits,
                                     size=x.shape[-2:],
                                     mode='bilinear',
                                     align_corners=False)

        # segmento principal
        seg_logits = self.segmentation_head(d1)
        combined = seg_logits + 0.3 * small_logits

        if self.training:
            aux3 = F.interpolate(self.aux_head_3(d3),
                                 size=x.shape[-2:],
                                 mode='bilinear',
                                 align_corners=False)
            aux2 = F.interpolate(self.aux_head_2(d2),
                                 size=x.shape[-2:],
                                 mode='bilinear',
                                 align_corners=False)
            return combined, aux3, aux2, small_logits

        return combined