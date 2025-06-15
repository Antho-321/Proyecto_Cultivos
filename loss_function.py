import torch
import torch.nn as nn
import torch.nn.functional as F
# --------------------------------------------------------------------
#  ❚❚❚  FOCAL LOSS  ❚❚❚
# --------------------------------------------------------------------
class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss tal como la describen Lin et al. (2017).
    • gamma:  cuánto penaliza aciertos fáciles  (habitual = 2.0).
    • alpha:  ponderación por clase   →  tensor [C]  (útil p/ clases raras).
    """
    def __init__(self, gamma: float = 2.0,
                 alpha: torch.Tensor | None = None,
                 reduction: str = "mean"):
        super().__init__()
        self.gamma      = gamma
        self.alpha      = alpha              # tensor en el mismo device que logits
        self.reduction  = reduction

    # --------------------------------------------------------------
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits : (N, C, H, W) – sin softmax
        targets: (N, H, W)    – enteros [0 … C-1]
        """
        n, c, h, w = logits.shape
        # log-probabilidades y probabilidades con softmax estable
        log_pt = torch.nn.functional.log_softmax(logits, dim=1)           # (N,C,H,W)
        pt     = log_pt.exp()                                             # (N,C,H,W)

        # extraemos p_t y log(p_t) de la clase objetivo → (N,H,W)
        targets_flat = targets.view(n, 1, h, w)
        log_pt = log_pt.gather(1, targets_flat).squeeze(1)
        pt     = pt    .gather(1, targets_flat).squeeze(1)

        loss = -(1 - pt) ** self.gamma * log_pt        # (N,H,W)

        if self.alpha is not None:
            # alpha debe ser un tensor de longitud C
            at = self.alpha[targets]                   # (N,H,W)
            loss = at * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss                                     # 'none'

# --------------------------------------------------------------------
#  ❚❚❚  DICE LOSS  ❚❚❚
# --------------------------------------------------------------------
class DiceLoss(nn.Module):
    """
    Dice Loss multiclase (promedio macro).  
    Devuelve 1 − Dice, por lo que **minimizarla maximiza el Dice**.
    """
    def __init__(self, smooth: float = 1e-6, ignore_index: int | None = None):
        super().__init__()
        self.smooth        = smooth
        self.ignore_index  = ignore_index

    # --------------------------------------------------------------
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits : (N, C, H, W)
        targets: (N, H, W)
        """
        num_classes = logits.shape[1]
        # Probabilidades con softmax
        probs = torch.nn.functional.softmax(logits, dim=1)        # (N,C,H,W)
        # One-hot de la máscara   → (N,C,H,W)
        one_hot = torch.nn.functional.one_hot(targets, num_classes
                     ).permute(0, 3, 1, 2).float()

        if self.ignore_index is not None:
            # anulamos la contribución de la clase ignorada
            mask = targets != self.ignore_index                    # (N,H,W)
            probs = probs * mask.unsqueeze(1)
            one_hot = one_hot * mask.unsqueeze(1)

        # aplanamos (N,C,H,W) → (N,C,H*W)
        probs  = probs .flatten(2)
        one_hot = one_hot.flatten(2)

        intersection = (probs * one_hot).sum(-1)                   # (N,C)
        cardinality  = probs.sum(-1) + one_hot.sum(-1)             # (N,C)
        dice_per_cls = (2. * intersection + self.smooth) / (cardinality + self.smooth)

        # Macro-promedio sobre clases y lote
        loss = 1 - dice_per_cls.mean()
        return loss

# --------------------------------------------------------------------
#  ❚❚❚  COMBO LOSS   (Focal + Dice)  ❚❚❚
# --------------------------------------------------------------------
class ComboLoss(nn.Module):
    """
    Híbrido de Focal Loss y Dice Loss:
        L = λ_focal · Focal + λ_dice · Dice
    Ajusta λ_* según la importancia que quieras dar a cada término.
    """
    def __init__(self,
                 lambda_focal: float = 1.0,
                 lambda_dice:  float = 1.0,
                 focal_gamma:   float = 2.0,
                 focal_alpha:   torch.Tensor | None = None):
        super().__init__()
        self.lambda_focal = lambda_focal
        self.lambda_dice  = lambda_dice
        self.focal = FocalLoss(gamma=focal_gamma, alpha=focal_alpha, reduction="mean")
        self.dice  = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        lf = self.focal(logits, targets)
        ld = self.dice (logits, targets)
        return self.lambda_focal * lf + self.lambda_dice * ld
