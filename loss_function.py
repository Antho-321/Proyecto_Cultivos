import torch
class AsymFocalTverskyLoss(torch.nn.Module):                                      # NEW
    def __init__(self, alpha=0.3, beta=0.7,
                 gamma_neg=1.0, gamma_pos=2.0, class_weights=None):
        super().__init__()
        self.alpha, self.beta = alpha, beta
        self.gneg, self.gpos  = gamma_neg, gamma_pos
        self.w = class_weights

    def forward(self, logits, gt):
        num_cls = logits.shape[1]
        gt_onehot = torch.nn.functional.one_hot(gt, num_cls).permute(0,3,1,2)
        prob = torch.softmax(logits, dim=1)

        TP = (prob * gt_onehot).sum((0,2,3))
        FP = (prob * (1-gt_onehot)).sum((0,2,3))
        FN = ((1-prob) * gt_onehot).sum((0,2,3))

        tversky = (TP + 1e-6) / (TP + self.alpha*FP + self.beta*FN + 1e-6)

        focal = torch.where(gt_onehot.sum((0,2,3))==0,
                            (1-tversky)**self.gneg,
                            (1-tversky)**self.gpos)

        if self.w is not None:
            focal = focal * self.w.to(focal.device)

        return focal.mean()
