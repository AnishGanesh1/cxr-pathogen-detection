import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss solves the class imbalance problem by down-weighting the loss 
    assigned to well-classified (easy) examples, forcing the model to focus 
    on the hard, rare diseases (like Pulmonary Fibrosis).
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # F.binary_cross_entropy_with_logits combines Sigmoid and BCE into one perfectly stable layer
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Calculate pt (the probability of the true class)
        # We know BCE loss equals -log(pt). Therefore: pt = exp(-BCE)
        pt = torch.exp(-bce_loss) 
        
        # Standard focal loss formula
        focal_loss = self.alpha * (1 - pt)**self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss
