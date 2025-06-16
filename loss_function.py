import torch
import torch.nn as nn
# ======================================================================
#  PÉRDIDAS BASADAS EN TVERSKY   (multi-clase, logits como entrada)
# ======================================================================

class TverskyLoss(nn.Module):
    """
    L_Tversky = 1 − TI,  donde
        TI = (TP) / (TP + α·FP + β·FN)

    • Soporta batch y múltiples clases.
    • Recibe logits (NO softmax) en `pred`.
    • target debe ser (N,H,W) con ints [0..C-1].
    """
    def __init__(
        self,
        n_classes: int,
        alpha: float = 0.3,
        beta: float  = 0.7,
        smooth: float = 1e-6,
        class_weights: torch.Tensor | None = None   # (C,) o None
    ):
        super().__init__()
        self.n_classes     = n_classes
        self.alpha         = alpha
        self.beta          = beta
        self.smooth        = smooth
        self.register_buffer(
            "class_weights",
            class_weights if class_weights is not None else torch.ones(n_classes)
        )

    # ------------------------------------------------------------------
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred: (N,C,H,W) logits
        # target: (N,H,W)  ints
        N, C, H, W = pred.shape
        pred_soft  = torch.softmax(pred, dim=1)     # (N,C,H,W)

        # One-hot y reshape a (N,C,H,W)
        target_1h = torch.nn.functional.one_hot(target, num_classes=C) \
                                     .permute(0,3,1,2).float()

        # TP, FP, FN por clase
        dims = (0, 2, 3)
        TP = (pred_soft * target_1h).sum(dim=dims)
        FP = (pred_soft * (1 - target_1h)).sum(dim=dims)
        FN = ((1 - pred_soft) * target_1h).sum(dim=dims)

        TI = (TP + self.smooth) / (
              TP + self.alpha*FP + self.beta*FN + self.smooth
        )                                           # (C,)

        # Promedio ponderado sobre clases
        loss = (1 - TI) * self.class_weights
        return loss.mean()


class FocalTverskyLoss(TverskyLoss):
    """
    L_FocalTversky = (1 − TI) ^ γ         (γ > 1 ↑ penaliza más FN)
    """
    def __init__(
        self,
        n_classes: int,
        alpha: float = 0.3,
        beta: float  = 0.7,
        gamma: float = 2.0,
        smooth: float = 1e-6,
        class_weights: torch.Tensor | None = None
    ):
        super().__init__(n_classes, alpha, beta, smooth, class_weights)
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        N, C, H, W = pred.shape
        pred_soft  = torch.softmax(pred, dim=1)
        target_1h  = torch.nn.functional.one_hot(target, num_classes=C) \
                                     .permute(0,3,1,2).float()

        TP = (pred_soft * target_1h).sum(dim=(0,2,3))
        FP = (pred_soft * (1 - target_1h)).sum(dim=(0,2,3))
        FN = ((1 - pred_soft) * target_1h).sum(dim=(0,2,3))

        TI = (TP + self.smooth) / (
              TP + self.alpha*FP + self.beta*FN + self.smooth
        )

        loss = (1 - TI) ** self.gamma
        loss = loss * self.class_weights
        return loss.mean()