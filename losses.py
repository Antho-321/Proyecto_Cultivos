import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, eps: float = 1e-7):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # y_pred: [B, C, H, W], probabilidades en [0,1]
        # y_true: [B, H, W] (enteros) o [B, C, H, W] (one-hot)
        if y_true.dim() == y_pred.dim() - 1:
            # convertir enteros a one-hot
            y_true = F.one_hot(y_true.long(), num_classes=y_pred.size(1)) \
                       .permute(0, 3, 1, 2) \
                       .float()
        else:
            y_true = y_true.float()

        # evitar log(0)
        y_pred = y_pred.clamp(self.eps, 1.0 - self.eps)

        # cross-entropy por píxel y clase
        ce = -y_true * torch.log(y_pred)

        # término focal
        loss = self.alpha * (1.0 - y_pred).pow(self.gamma) * ce

        # devuelve [B, C, H, W] (no se agrupa aquí)
        return loss

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # ambas con forma [B, C, H, W], one-hot y probabilidades
        # aplanar
        y_true_f = y_true.reshape(y_true.size(0), -1)
        y_pred_f = y_pred.reshape(y_pred.size(0), -1)

        intersection = (y_true_f * y_pred_f).sum(dim=1)
        dice_score = (2.0 * intersection + self.smooth) / (
            y_true_f.sum(dim=1) + y_pred_f.sum(dim=1) + self.smooth
        )
        return 1.0 - dice_score.mean()

class CombinedLoss(nn.Module):
    def __init__(
        self,
        class_weights: list[float] | None = None,
        gamma: float = 2.0,
        alpha: float = 0.75,
        dice_weight: float = 1.5,
    ):
        super().__init__()
        self.focal = FocalLoss(gamma=gamma, alpha=alpha)
        self.dice  = DiceLoss()
        if class_weights is None:
            class_weights = [1, 1, 1, 1, 2, 1]
        # shape [1, C, 1, 1] para broadcasting
        self.register_buffer(
            "class_weights",
            torch.tensor(class_weights, dtype=torch.float32)
                 .view(1, -1, 1, 1)
        )
        self.dice_weight = dice_weight

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # convertir y_true a one-hot si hace falta
        if y_true.dim() == y_pred.dim() - 1:
            y_true_onehot = F.one_hot(y_true.long(), num_classes=y_pred.size(1)) \
                             .permute(0, 3, 1, 2) \
                             .float()
        else:
            y_true_onehot = y_true.float()

        # pérdida focal sin reducción
        f_loss = self.focal(y_pred, y_true_onehot)  # [B, C, H, W]

        # aplicar pesos de clase
        weighted_f = f_loss * self.class_weights

        # sumar sobre la dimensión de clases y promediar
        f_loss_per_pixel = weighted_f.sum(dim=1)     # [B, H, W]
        f_loss_mean      = f_loss_per_pixel.mean()   # escalar

        # pérdida dice
        d_loss = self.dice(y_pred, y_true_onehot)

        return f_loss_mean + self.dice_weight * d_loss