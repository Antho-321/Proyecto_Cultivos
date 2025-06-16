import torch
class AsymFocalTverskyLoss(torch.nn.Module):                                      # NEW
    def __init__(self, class_weights=None, alpha=0.5, beta=0.5,
                 gamma=0.75, eps=1e-4):          # ← añadido eps
        super().__init__()
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.register_buffer("class_weights",
                             class_weights if class_weights is not None
                             else torch.ones(6))
        self.eps = eps 

    def forward(self, logits, gt):
        """
        logits : Tensor (N, C, H, W)  — salida sin activar del modelo
        gt     : Tensor (N, H, W)     — máscara con enteros [0‥C-1]
        """
        num_cls     = logits.shape[1]

        # 1) One-hot de la GT  →  (N, C, H, W)
        gt_onehot   = torch.nn.functional.one_hot(gt, num_cls).permute(0, 3, 1, 2)

        # 2) Probabilidades predichas
        prob        = torch.softmax(logits, dim=1)

        # 3) Estadísticos por clase
        TP = (prob * gt_onehot).sum((0, 2, 3))
        FP = (prob * (1 - gt_onehot)).sum((0, 2, 3))
        FN = ((1 - prob) * gt_onehot).sum((0, 2, 3))

        # 4) Índice de Tversky con ε configurable
        tversky = (TP + self.eps) / (TP + self.alpha * FP + self.beta * FN + self.eps)

        # 5) Focal-Tversky asimétrico
        focal = torch.where(
            gt_onehot.sum((0, 2, 3)) == 0,          # clase ausente en el batch
            (1 - tversky) ** self.gneg,             # γ_neg
            (1 - tversky) ** self.gpos              # γ_pos
        )

        # 6) Ponderación opcional por clase
        if self.w is not None:
            focal = focal * self.w.to(focal.device)

        return focal.mean()
