import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalTverskyLoss(nn.Module):
    """Combined Focal + Tversky loss for better handling of class imbalance"""
    def __init__(self, alpha=0.7, beta=0.3, gamma=2.0, smooth=1e-6):
        super().__init__()
        self.alpha = alpha  # Weight for false negatives
        self.beta = beta    # Weight for false positives  
        self.gamma = gamma  # Focal parameter
        self.smooth = smooth
        
    def forward(self, pred, target):
        # Convert to one-hot
        num_classes = pred.shape[1]
        target_one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
        pred_soft = F.softmax(pred, dim=1)
        
        # Compute Tversky for each class
        tversky_loss = 0
        for c in range(num_classes):
            pred_c = pred_soft[:, c]
            target_c = target_one_hot[:, c]
            
            tp = (pred_c * target_c).sum()
            fp = (pred_c * (1 - target_c)).sum()
            fn = ((1 - pred_c) * target_c).sum()
            
            tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
            focal_tversky = (1 - tversky) ** self.gamma
            tversky_loss += focal_tversky
            
        return tversky_loss / num_classes