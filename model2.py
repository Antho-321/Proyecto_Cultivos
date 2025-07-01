import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# ----------------------------------------------------------------------
# Optional – keep only ChannelAttention (light) for deepest skip
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return x * self.fc(self.avg_pool(x))

# ----------------------------------------------------------------------
# Much lighter ASPP
class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch=128, rates=(6, 12)):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False),
                          nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)),
            *[nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=r, dilation=r, bias=False),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)) for r in rates],
            nn.Sequential(nn.AdaptiveAvgPool2d(1),
                          nn.Conv2d(in_ch, out_ch, 1, bias=False),
                          nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        ])
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * len(self.blocks), out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Dropout(0.5)                       # stronger dropout
        )

    def forward(self, x):
        feats = [blk(x) if i else blk(x) for i, blk in enumerate(self.blocks)]
        feats[-1] = F.interpolate(feats[-1], size=x.shape[-2:], mode='bilinear', align_corners=False)
        return self.project(torch.cat(feats, dim=1))

# ----------------------------------------------------------------------
# Slim decoder block with optional attention only on last skip
class DecoderBlock(nn.Module):
    def __init__(self, skip_ch, up_ch, out_ch, use_attn=False):
        super().__init__()
        self.up = nn.ConvTranspose2d(up_ch, up_ch, 2, stride=2)
        self.attn = ChannelAttention(skip_ch) if use_attn else nn.Identity()
        self.fuse = nn.Sequential(
            nn.Conv2d(skip_ch + up_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, skip, up):
        up = self.up(up)
        skip = self.attn(skip)
        return self.fuse(torch.cat([up, skip], dim=1))

# ----------------------------------------------------------------------
# Main model
class LiteDeepLabFPN(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            'mobilenetv3_large_100',
            features_only=True, pretrained=pretrained,
            out_indices=(0, 1, 2, 3)             # Strides 2,4,8,16
        )
        chs = self.backbone.feature_info.channels()   # ≈[16, 24, 40, 112]

        self.aspp = ASPP(chs[3], out_ch=128)

        self.dec3 = DecoderBlock(chs[2], 128, 64, use_attn=True)   # deepest skip keeps attention
        self.dec2 = DecoderBlock(chs[1], 64, 32)
        self.dec1 = DecoderBlock(chs[0], 32, 24)

        self.head = nn.Sequential(
            nn.Conv2d(24, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(32, num_classes, 1)
        )
        self.up_final = nn.UpsamplingBilinear2d(scale_factor=2)  # stride-1 output

    def forward(self, x):
        feats = self.backbone(x)          # 4 scales
        x = self.aspp(feats[3])
        x = self.dec3(feats[2], x)
        x = self.dec2(feats[1], x)
        x = self.dec1(feats[0], x)
        return self.up_final(self.head(x))