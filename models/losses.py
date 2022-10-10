import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=2, gamma=0.8):
        inputs = torch.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        bce = F.binary_cross_entropy(inputs, targets, reduction='mean')
        bce_exp = torch.exp(-bce)
        focal_loss = alpha * (1 - bce_exp)**gamma * bce

        return focal_loss


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, targets, inputs, smooth=1):
        inputs = torch.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()

        dice = (2.*intersection + smooth) / \
            (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()
        self.dice = DiceLoss()

    def forward(self, inputs, targets, smooth=1):
        dice = self.dice(inputs, targets, smooth)

        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        bce = F.binary_cross_entropy(inputs, targets, reduction='mean')

        return dice + bce


class DiceFocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceFocalLoss, self).__init__()
        self.dice = DiceLoss()
        self.focal = FocalLoss()

    def forward(self, inputs, targets, smooth=1, alpha=2, gamma=0.8):
        dice = self.dice(inputs, targets, smooth)
        focal = self.focal(inputs, targets, alpha, gamma)

        return dice + focal
