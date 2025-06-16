import torch
import torch.nn as nn
import torch.nn.functional as F

class AsymFocalTverskyLoss(nn.Module):
    """
    Implementación con:
      • α, β           — ponderación FP/FN del Tversky
      • g_pos, g_neg   — γ asimétrico (clase presente / ausente)
      • w              — pesos por clase (Tensor[C]) opcional
      • eps            — término de estabilidad numérica
    """
    def __init__(self,
                 class_weights=None,        # → w
                 alpha: float = 0.5,
                 beta : float = 0.5,
                 g_pos: float = 0.75,
                 g_neg: float = 1.50,
                 eps  : float = 1e-4):
        super().__init__()
        self.alpha, self.beta = alpha, beta
        self.gpos,  self.gneg = g_pos, g_neg     # ← NUEVO
        # Registramos los pesos como buffer para que sigan al dispositivo
        w = torch.ones(6) if class_weights is None else class_weights
        self.register_buffer("w", w.float())     # ← NUEVO (se llamará self.w)
        self.eps = eps                           # ← ya lo tenías

    # ------------------------------------------------------------------
    def forward(self, logits, gt):
        """
        logits : (N, C, H, W)  — scores sin activar
        gt     : (N, H, W)     — etiquetas enteras [0..C-1]
        """
        C          = logits.size(1)
        gt_1hot    = F.one_hot(gt, C).permute(0, 3, 1, 2).float()
        prob       = F.softmax(logits, dim=1)

        TP = (prob * gt_1hot).sum((0, 2, 3))
        FP = (prob * (1 - gt_1hot)).sum((0, 2, 3))
        FN = ((1 - prob) * gt_1hot).sum((0, 2, 3))

        tversky = (TP + self.eps) / (TP + self.alpha * FP + self.beta * FN + self.eps)

        # Focal-Tversky asimétrico
        focal = torch.where(
            gt_1hot.sum((0, 2, 3)) == 0,   # clase ausente
            (1 - tversky) ** self.gneg,
            (1 - tversky) ** self.gpos
        )

        loss = (self.w.to(focal.device) * focal).mean()
        return loss
