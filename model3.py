import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# 1) Depthwise separable conv
# -------------------------
class SeparableConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size, stride, padding,
            groups=in_ch, bias=False
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)


# -------------------------
# 2) Xception “ligero” backbone
#    (Entry/Middle/Exit flows con avg pooling extra)
# -------------------------
class XceptionLite(nn.Module):
    def __init__(self):
        super().__init__()
        # Entry flow
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.entry_blocks = nn.ModuleList([
            nn.Sequential(
                SeparableConv2d(64, 128),
                SeparableConv2d(128, 128),
                nn.MaxPool2d(3, stride=2, padding=1)
            ),
            nn.Sequential(
                SeparableConv2d(128, 256),
                SeparableConv2d(256, 256),
                nn.MaxPool2d(3, stride=2, padding=1)
            ),
            nn.Sequential(
                SeparableConv2d(256, 728),
                SeparableConv2d(728, 728),
                nn.MaxPool2d(3, stride=2, padding=1)
            ),
        ])
        # Middle flow (16 × triple‐SeparableConv)
        mid = []
        for _ in range(16):
            mid += [
                SeparableConv2d(728, 728),
                SeparableConv2d(728, 728),
                SeparableConv2d(728, 728)
            ]
        self.middle_flow = nn.Sequential(*mid)
        # Exit flow + avg‐pool extra
        self.exit_flow = nn.Sequential(
            SeparableConv2d(728, 1024),
            nn.AvgPool2d(2, stride=2),
            SeparableConv2d(1024, 1536),
            nn.AvgPool2d(2, stride=2),
            SeparableConv2d(1536, 1536),
            nn.AvgPool2d(2, stride=2),
            SeparableConv2d(1536, 2048),
        )

    def forward(self, x):
        x = self.conv1(x)
        low_level = x
        for b in self.entry_blocks:
            x = b(x)
        x = self.middle_flow(x)
        x = self.exit_flow(x)
        return x, low_level


# -------------------------
# 3) ASPP multitasas + CBAM corregido
# -------------------------
class ASPP_CBAM(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        rates = [1, 3, 9, 12]
        # ASPP convolutions
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_ch, out_ch,
                    1 if r == 1 else 3,
                    padding=0 if r == 1 else r,
                    dilation=r,
                    bias=False
                ),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
            for r in rates
        ])
        # reduce 4*out_ch → out_ch
        self.reduce = nn.Sequential(
            nn.Conv2d(len(rates) * out_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        # channel attention (CBAM)
        self.ca_avg = nn.AdaptiveAvgPool2d(1)
        self.ca_max = nn.AdaptiveMaxPool2d(1)
        self.ca_fc  = nn.Sequential(
            nn.Conv2d(out_ch, out_ch // 8, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch // 8, out_ch, 1, bias=False),
            nn.Sigmoid()
        )
        # spatial attention (CBAM)
        self.sa_conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sa_sig  = nn.Sigmoid()

    def forward(self, x):
        # ASPP feature maps
        feats = [c(x) for c in self.convs]
        x = torch.cat(feats, dim=1)    # [B, 4*out_ch, H, W]
        x = self.reduce(x)             # [B, out_ch, H, W]

        # channel attention
        avg = self.ca_avg(x)
        maxp = self.ca_max(x)
        ca = self.ca_fc(avg + maxp)
        x = x * ca

        # spatial attention
        avgc = torch.mean(x, dim=1, keepdim=True)
        maxc, _ = torch.max(x, dim=1, keepdim=True)
        sa = self.sa_sig(self.sa_conv(torch.cat([avgc, maxc], dim=1)))
        return x * sa


# -------------------------
# 4) Decoder con GAU
# -------------------------
class GAU(nn.Module):
    def __init__(self, low_ch, high_ch, out_ch):
        super().__init__()
        self.conv_low = nn.Conv2d(low_ch, out_ch, 3, padding=1, bias=False)
        self.pool     = nn.AdaptiveAvgPool2d(1)
        self.fc       = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, high, low):
        low_proj = self.conv_low(low)
        w = self.pool(low_proj)
        w = self.fc(w)
        low_w = low_proj * w
        high_up = F.interpolate(
            high,
            size=low_w.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        return low_w + high_up


# -------------------------
# 5) Improved DeepLabV3+ completo
# -------------------------
class ImprovedDeepLabV3Plus(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.backbone  = XceptionLite()
        self.aspp_cbam = ASPP_CBAM(in_ch=2048, out_ch=256)
        self.decoder   = GAU(low_ch=64, high_ch=256, out_ch=256)
        self.classifier = nn.Conv2d(256, n_classes, 1)

    def forward(self, x):
        high, low = self.backbone(x)
        x = self.aspp_cbam(high)
        x = self.decoder(x, low)
        x = self.classifier(x)
        return F.interpolate(
            x,
            scale_factor=4,
            mode='bilinear',
            align_corners=False
        )
