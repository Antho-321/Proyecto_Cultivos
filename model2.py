import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# =================================================================================
# 1. MÓDULO ASPP
# =================================================================================
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        size = x.shape[-2:]
        x = super().forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256, atrous_rates=(6, 12, 18)):
        super().__init__()
        modules = [
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        ]
        for rate in atrous_rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))
        modules.append(ASPPPooling(in_channels, out_channels))
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs)*out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        x = torch.cat(res, dim=1)
        return self.project(x)


# =================================================================================
# 2. DECODER SIMPLE
# =================================================================================
class Decoder(nn.Module):
    def __init__(self, low_level_in, num_classes):
        super().__init__()
        self.reduce_low = nn.Sequential(
            nn.Conv2d(low_level_in, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(48 + 256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, low_level_feat, x):
        low = self.reduce_low(low_level_feat)
        x = F.interpolate(x, size=low.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, low], dim=1)
        x = self.fuse(x)
        return self.classifier(x)


# =================================================================================
# 3. ARQUITECTURA PRINCIPAL
# =================================================================================
class DeepLabV3Plus_EfficientNetV2S(nn.Module):
    def __init__(self, input_shape=(128,128,3), num_classes=1):
        super().__init__()
        # Backbone EfficientNetV2-S
        self.backbone = timm.create_model(
            'tf_efficientnetv2_s.in21k',
            pretrained=True,
            features_only=True,
            out_indices=(1, 2, 3, 4)  # stages: low-level at idx 1, high-level at idx 4
        )
        backbone_channels = self.backbone.feature_info.channels()
        low_level_ch     = backbone_channels[0]    # corresponde a out_indices[0] == 1
        high_level_ch    = backbone_channels[-1]   # corresponde a out_indices[-1] == 4

        # ASPP y Decoder
        self.aspp = ASPP(in_channels=high_level_ch, out_channels=256, atrous_rates=(6,12,18))
        self.decoder = Decoder(low_level_in=low_level_ch, num_classes=num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        low_level_feat   = feats[0]
        high_level_feat  = feats[-1]

        x = self.aspp(high_level_feat)
        x = self.decoder(low_level_feat, x)
        x = F.interpolate(x, size=x.shape[-2]*4, mode='bilinear', align_corners=False)  # up to original
        return torch.sigmoid(x)


# # Ejemplo de uso:
# model = DeepLabV3Plus_EfficientNetV2S(input_shape=(128,128,3), num_classes=1)
# print(model)
# input_tensor = torch.randn(1,3,128,128)
# output = model(input_tensor)
# print(output.shape)  # torch.Size([1, 1, 128, 128])
