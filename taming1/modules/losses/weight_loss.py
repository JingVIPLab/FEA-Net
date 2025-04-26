import math
import torch
from torch import autograd as autograd
from torch import nn as nn
from torch.nn import functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftCrossEntropy(nn.Module):
    def __init__(self, loss_weight=1.0, adaptive_weighting=True, reduction='mean'):
        super(SoftCrossEntropy, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')
        self.adaptive_weighting = adaptive_weighting
        self.reduction = reduction
        self.loss_weight = loss_weight


    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Probabilistic target tensor.
            weight (Tensor, optional): of shape (C). Weights to multiply the loss per class. Default: None
        """
        N, C, H, W = pred.shape
        cum_losses = pred.new_zeros((N, H, W))

        if self.adaptive_weighting:
            class_weights = self.compute_adaptive_weights(target)

        for y in range(C):
            target_temp = pred.new_full((N, H, W,), y, dtype=torch.long)
            y_loss = F.cross_entropy(pred, target_temp, reduction="mean")

            if self.adaptive_weighting:
                y_loss = y_loss * class_weights[y]
            elif weight is not None:
                y_loss = y_loss * weight[y]

            cum_losses += target[:, y, :, :].float() * y_loss

        cum_losses *= self.loss_weight

        if cum_losses.mean() < 0:
            cum_losses = abs(cum_losses)

        if self.reduction == "none":
            return cum_losses
        elif self.reduction == "mean":
            return cum_losses.mean()
        elif self.reduction == "sum":
            return cum_losses.sum()
        else:
            raise ValueError("Keyword 'reduction' must be one of ['none', 'mean', 'sum']")

    def compute_adaptive_weights(self, target):
        """
        Compute adaptive weights based on target distribution.
        Args:
            target (Tensor): of shape (N, C, H, W). Probabilistic target tensor.
        Returns:
            class_weights (Tensor): of shape (C). Adaptive weights for each class.
        """
        class_weights = torch.mean(target, dim=(0, 2, 3))  # Compute mean probability for each class
        class_weights = 1.0 / (class_weights + 1e-10)  # Inverse mean probability
        class_weights /= torch.sum(class_weights)  # Normalize weights
        return class_weights




