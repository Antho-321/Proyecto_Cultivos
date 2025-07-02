import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from coordatt import CoordAtt  # pip install coordinate-attention-pytorch

class ImprovedEfficientNetV2(nn.Module):
    """
    CNN branch: Improved EfficientNet-V2 with five output stages E0–E4
    (downscales 1/2,1/4,1/8,1/16,1/32) :contentReference[oaicite:0]{index=0}
    """
    def __init__(self, in_channels=4, variant='efficientnetv2_s', out_indices=(0,1,2,3,4)):
        super().__init__()
        # load a pretrained EfficientNet-V2 and adapt first conv for 4-channel input
        self.backbone = timm.create_model(variant, pretrained=True, features_only=True, out_indices=out_indices)
        # patch first conv weight to accept 4 channels
        conv0 = self.backbone.conv_stem
        self.backbone.conv_stem = nn.Conv2d(in_channels, conv0.out_channels,
                                            kernel_size=conv0.kernel_size, stride=conv0.stride,
                                            padding=conv0.padding, bias=conv0.bias is not None)
        with torch.no_grad():
            # initialize new channel by averaging the RGB weights
            self.backbone.conv_stem.weight[:, :3] = conv0.weight
            self.backbone.conv_stem.weight[:, 3] = conv0.weight.mean(dim=1)

    def forward(self, x):
        # returns list [E0, E1, E2, E3, E4]
        return self.backbone(x)


class CSWinBackbone(nn.Module):
    """
    Transformer branch: CSwin-Tiny producing T1–T4
    (downscales 1/4,1/8,1/16,1/32) :contentReference[oaicite:1]{index=1}
    """
    def __init__(self, variant='cswin_tiny_224', in_channels=4, out_indices=(1,2,3,4)):
        super().__init__()
        # CSwin typically expects 3-channel input; embed 4-channel similarly
        self.backbone = timm.create_model(variant, pretrained=True, features_only=True, out_indices=out_indices)
        # replace patch embedding to accept 4 channels
        pe = self.backbone.patch_embed.proj
        self.backbone.patch_embed.proj = nn.Conv2d(in_channels, pe.out_channels,
                                                   kernel_size=pe.kernel_size, stride=pe.stride,
                                                   padding=pe.padding, bias=pe.bias is not None)
        with torch.no_grad():
            self.backbone.patch_embed.proj.weight[:, :3] = pe.weight
            self.backbone.patch_embed.proj.weight[:, 3] = pe.weight.mean(dim=1)

    def forward(self, x):
        # returns list [T1, T2, T3, T4]
        return self.backbone(x)


class CAFM(nn.Module):
    """
    Coordinate Attention Fusion Module (CAFM) :contentReference[oaicite:2]{index=2}
    """
    def __init__(self, channels, reduction=32):
        super().__init__()
        # F1: 1×1 conv to unify channel dims
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1)
        # coordinate attention
        self.ca = CoordAtt(channels, reduction)

    def forward(self, xE, xT):
        # apply F1 + CA to each branch
        yE = self.ca(self.conv1(xE))
        yT = self.ca(self.conv1(xT))
        # fused feature yET
        yET = self.ca(self.conv1(torch.cat([xE, xT], dim=1)))
        # compute pixel-wise weights
        w = self.conv1(torch.cat([yE, yT], dim=1))
        w = F.softmax(w, dim=1)
        # split into two
        wE, wT = torch.chunk(w, 2, dim=1)
        # weighted sum and residual
        fused = yE * wE + yT * wT + yET
        # final CA
        return self.ca(fused)


class FPNHead(nn.Module):
    """
    Decoder: Multiscale FPNHead followed by CA and upsampling 
    """
    def __init__(self, in_channels_list, out_channels=256, num_classes=1):
        super().__init__()
        # lateral 1×1 convs
        self.laterals = nn.ModuleList([
            nn.Conv2d(c, out_channels, 1) for c in in_channels_list
        ])
        # smooth 3×3 convs
        self.smooths = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels_list
        ])
        # final CA
        self.final_ca = CoordAtt(len(in_channels_list)*out_channels, reduction=32)
        # classifier
        self.classifier = nn.Conv2d(len(in_channels_list)*out_channels, num_classes, kernel_size=1)

    def forward(self, feats):
        # feats: [ET1, ET2, ET3, ET4] from smallest to coarsest
        # build top-down path
        laterals = [l(feat) for l, feat in zip(self.laterals, feats)]
        for i in range(len(laterals)-1, 0, -1):
            laterals[i-1] = laterals[i-1] + F.interpolate(laterals[i], size=laterals[i-1].shape[-2:], mode='bilinear', align_corners=False)
        # apply smoothing
        outs = [s(l) for s, l in zip(self.smooths, laterals)]
        # upsample all to input resolution
        target_size = outs[0].shape[-2:]
        ups = [F.interpolate(o, size=target_size, mode='bilinear', align_corners=False) for o in outs]
        # concat and refine
        x = torch.cat(ups, dim=1)
        x = self.final_ca(x)
        x = self.classifier(x)
        # upsample to original input size if needed outside
        return x


class ParallelFusionNet(nn.Module):
    """
    Full network: CNN + Transformer branches, CAFM fusion at four scales,
    then FPNHead decoder (see overall Fig. 2) 
    """
    def __init__(self, num_classes=1):
        super().__init__()
        self.cnn = ImprovedEfficientNetV2(in_channels=4)
        self.trans = CSWinBackbone(in_channels=4)
        # channels of E1–E4 and T1–T4 are all out_channels of backbone; here assume 256
        ch = 256
        self.fusors = nn.ModuleList([CAFM(ch) for _ in range(4)])
        self.decoder = FPNHead([ch]*4, out_channels=ch, num_classes=num_classes)

    def forward(self, x):
        # extract local features E0–E4
        E = self.cnn(x)
        # extract global features T1–T4
        T = self.trans(x)
        # fuse at four scales: (E1,T1)...(E4,T4)
        fused = [self.fusors[i](E[i+1], T[i]) for i in range(4)]
        # decode
        seg = self.decoder(fused)
        # final upsample to input size
        return F.interpolate(seg, size=x.shape[-2:], mode='bilinear', align_corners=False)