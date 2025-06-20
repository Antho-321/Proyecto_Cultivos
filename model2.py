import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# =================================================================================
# 1. ENHANCED ATTENTION MODULE
# =================================================================================
class ChannelAttention(nn.Module):
    """Enhanced Channel Attention with ECA mechanism."""
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # ECA-style efficient channel attention
        kernel_size = int(abs((torch.log2(torch.tensor(in_channels, dtype=torch.float32)) + 1) / 2))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        
        self.eca = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        
        # Traditional squeeze-excitation as backup
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        
        # ECA branch
        y_avg = self.avg_pool(x).view(b, 1, c)
        y_eca = self.eca(y_avg).view(b, c, 1, 1)
        
        # SE branch
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        
        # Combine both mechanisms
        out = y_eca + avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    """Enhanced Spatial Attention with multi-scale context."""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        # Multi-scale spatial attention
        self.conv3x3 = nn.Conv2d(2, 1, 3, padding=1, bias=False)
        self.conv5x5 = nn.Conv2d(2, 1, 5, padding=2, bias=False)
        self.conv7x7 = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        
        # Attention weights for multi-scale fusion
        self.scale_weights = nn.Parameter(torch.ones(3))
        self.sigmoid = nn.Sigmoid()
        
        # Learnable channel weighting
        self.channel_weight = nn.Parameter(torch.tensor([1.0, 1.0]))

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Weighted combination of avg and max
        weighted_avg = avg_out * self.channel_weight[0]
        weighted_max = max_out * self.channel_weight[1]
        concat = torch.cat([weighted_avg, weighted_max], dim=1)
        
        # Multi-scale processing
        out3 = self.conv3x3(concat)
        out5 = self.conv5x5(concat)
        out7 = self.conv7x7(concat)
        
        # Weighted fusion
        weights = F.softmax(self.scale_weights, dim=0)
        out = weights[0] * out3 + weights[1] * out5 + weights[2] * out7
        
        return self.sigmoid(out)

class AttentionModule(nn.Module):
    """Enhanced Attention Module with residual connections."""
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(AttentionModule, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
        
        # Learnable fusion weights
        self.fusion_weight = nn.Parameter(torch.tensor([0.5, 0.5]))

    def forward(self, x):
        # Channel attention
        x_ca = x * self.channel_attention(x)
        
        # Spatial attention
        x_sa = x * self.spatial_attention(x)
        
        # Weighted fusion with residual
        weights = F.softmax(self.fusion_weight, dim=0)
        out = weights[0] * x_ca + weights[1] * x_sa + 0.1 * x  # Small residual
        
        return out

# =================================================================================
# 2. ENHANCED ASPP MODULE
# =================================================================================
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        super(ASPPConv, self).__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)  # Light dropout for regularization
        )

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class EnhancedASPP(nn.Module):
    """Enhanced ASPP with more scales and better fusion."""
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(EnhancedASPP, self).__init__()
        modules = []
        
        # 1x1 conv
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        
        # Dilated convolutions with enhanced rates
        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))
        
        # Global average pooling
        modules.append(ASPPPooling(in_channels, out_channels))
        
        self.convs = nn.ModuleList(modules)
        
        # Enhanced projection with attention
        total_channels = len(self.convs) * out_channels
        self.project = nn.Sequential(
            nn.Conv2d(total_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2)
        )
        
        # Self-attention for better feature fusion
        self.self_attention = nn.MultiheadAttention(out_channels, num_heads=8, batch_first=True)
        
    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        
        # Concatenate all features
        concat_features = torch.cat(res, dim=1)
        projected = self.project(concat_features)
        
        # Apply self-attention for better feature interaction
        b, c, h, w = projected.shape
        projected_flat = projected.view(b, c, -1).permute(0, 2, 1)  # (b, h*w, c)
        
        attended, _ = self.self_attention(projected_flat, projected_flat, projected_flat)
        attended = attended.permute(0, 2, 1).view(b, c, h, w)
        
        # Residual connection
        return projected + 0.1 * attended

# =================================================================================
# 3. ENHANCED DECODER BLOCK
# =================================================================================
class EnhancedDecoderBlock(nn.Module):
    """Enhanced Decoder Block with better feature fusion."""
    def __init__(self, in_channels_skip, in_channels_up, out_channels, use_attention=True):
        super(EnhancedDecoderBlock, self).__init__()
        
        # Learnable upsampling
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels_up, in_channels_up, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(in_channels_up),
            nn.ReLU(inplace=True)
        )
        
        # Channel alignment with 1x1 conv
        self.align_skip = nn.Conv2d(in_channels_skip, out_channels, 1, bias=False)
        self.align_up = nn.Conv2d(in_channels_up, out_channels, 1, bias=False)
        
        self.use_attention = use_attention
        if use_attention:
            self.attention = AttentionModule(in_channels=in_channels_skip, kernel_size=7)
        
        # Enhanced fusion block
        self.fusion_block = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
        
        # Feature enhancement
        self.feature_enhancement = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()
        )

    def forward(self, x_skip, x_up):
        # Upsample
        x_up = self.upsample(x_up)
        
        # Apply attention to skip connection
        if self.use_attention:
            x_skip = self.attention(x_skip)
        
        # Align channels
        x_skip_aligned = self.align_skip(x_skip)
        x_up_aligned = self.align_up(x_up)
        
        # Ensure spatial dimensions match
        if x_skip_aligned.shape[-2:] != x_up_aligned.shape[-2:]:
            x_up_aligned = F.interpolate(x_up_aligned, size=x_skip_aligned.shape[-2:], 
                                       mode='bilinear', align_corners=False)
        
        # Concatenate and fuse
        x_concat = torch.cat([x_skip_aligned, x_up_aligned], dim=1)
        x_fused = self.fusion_block(x_concat)
        
        # Feature enhancement
        enhancement = self.feature_enhancement(x_fused)
        x_enhanced = x_fused * enhancement
        
        # Residual connection
        return x_enhanced + x_skip_aligned

# =================================================================================
# 4. ENHANCED MAIN ARCHITECTURE
# =================================================================================
class EnhancedCloudDeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=1):
        super(EnhancedCloudDeepLabV3Plus, self).__init__()
        
        # Enhanced atrous rates for better multi-scale context
        atrous_rates = (3, 6, 12, 18)
        
        # Use a stronger backbone
        self.backbone = timm.create_model(
            'tf_efficientnetv2_s.in21k',
            pretrained=True,
            features_only=True,
            out_indices=(0, 1, 2, 3, 4)  # More feature levels
        )
        
        backbone_channels = self.backbone.feature_info.channels()
        # Typically: [24, 48, 80, 176, 304] for efficientnetv2_m
        
        # Enhanced ASPP
        self.aspp = EnhancedASPP(
            in_channels=backbone_channels[-1], 
            atrous_rates=atrous_rates, 
            out_channels=320
        )
        
        # Enhanced decoder with more levels
        self.decoder_block4 = EnhancedDecoderBlock(
            in_channels_skip=backbone_channels[3],
            in_channels_up=320,
            out_channels=176
        )
        self.decoder_block3 = EnhancedDecoderBlock(
            in_channels_skip=backbone_channels[2],
            in_channels_up=176,
            out_channels=128
        )
        self.decoder_block2 = EnhancedDecoderBlock(
            in_channels_skip=backbone_channels[1],
            in_channels_up=128,
            out_channels=64
        )
        self.decoder_block1 = EnhancedDecoderBlock(
            in_channels_skip=backbone_channels[0],
            in_channels_up=64,
            out_channels=48
        )
        
        # Enhanced segmentation head with pyramid pooling
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(48, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(16, num_classes, 1)
        )
        
        # Auxiliary heads for deep supervision
        self.aux_head_4 = nn.Conv2d(176, num_classes, 1)
        self.aux_head_3 = nn.Conv2d(128, num_classes, 1)
        self.aux_head_2 = nn.Conv2d(64, num_classes, 1)
        
        # Boundary refinement module
        self.boundary_refinement = nn.Sequential(
            nn.Conv2d(num_classes, num_classes, 3, padding=1, bias=False),
            nn.BatchNorm2d(num_classes),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_classes, num_classes, 1)
        )
        
        # Final upsampling
        self.final_upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        input_size = x.shape[-2:]
        features = self.backbone(x)
        
        # ASPP on deepest features
        aspp_output = self.aspp(features[-1])
        
        # Progressive decoding
        decoder_out4 = self.decoder_block4(features[3], aspp_output)
        decoder_out3 = self.decoder_block3(features[2], decoder_out4)
        decoder_out2 = self.decoder_block2(features[1], decoder_out3)
        decoder_out1 = self.decoder_block1(features[0], decoder_out2)
        
        # Main segmentation output
        logits = self.segmentation_head(decoder_out1)
        
        # Boundary refinement
        refined_logits = self.boundary_refinement(logits)
        
        # Final upsampling
        final_logits = self.final_upsample(refined_logits)
        
        # Ensure output matches input size
        if final_logits.shape[-2:] != input_size:
            final_logits = F.interpolate(final_logits, size=input_size, 
                                       mode='bilinear', align_corners=False)
        
        if self.training:
            # Auxiliary outputs for deep supervision
            aux4 = F.interpolate(self.aux_head_4(decoder_out4), size=input_size, 
                               mode='bilinear', align_corners=False)
            aux3 = F.interpolate(self.aux_head_3(decoder_out3), size=input_size, 
                               mode='bilinear', align_corners=False)
            aux2 = F.interpolate(self.aux_head_2(decoder_out2), size=input_size, 
                               mode='bilinear', align_corners=False)
            
            return final_logits, aux4, aux3, aux2
        
        return final_logits

# =================================================================================
# 5. TRAINING UTILITIES FOR BETTER mIoU
# =================================================================================
class CombinedLoss(nn.Module):
    """Combined loss function for better boundary handling."""
    def __init__(self, alpha=0.25, gamma=2.0, aux_weight=0.4):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.aux_weight = aux_weight
        self.bce = nn.BCEWithLogitsLoss()
        
    def focal_loss(self, pred, target):
        """Focal loss for handling class imbalance."""
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()
    
    def dice_loss(self, pred, target, smooth=1e-5):
        """Dice loss for better overlap."""
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()
    
    def forward(self, pred, target, aux_preds=None):
        """Combined loss calculation."""
        main_loss = 0.5 * self.focal_loss(pred, target) + 0.5 * self.dice_loss(pred, target)
        
        total_loss = main_loss
        
        if aux_preds is not None and self.training:
            aux_loss = 0
            for aux_pred in aux_preds:
                aux_loss += 0.5 * self.focal_loss(aux_pred, target) + 0.5 * self.dice_loss(aux_pred, target)
            total_loss += self.aux_weight * aux_loss / len(aux_preds)
        
        return total_loss

# Usage example:
# model = EnhancedCloudDeepLabV3Plus(num_classes=1)
# criterion = CombinedLoss()
# 
# For training:
# outputs = model(images)  # Returns (main_output, aux1, aux2, aux3) during training
# loss = criterion(outputs[0], targets, outputs[1:])
#
# For inference:
# model.eval()
# with torch.no_grad():
#     output = model(images)  # Returns only main output during inference