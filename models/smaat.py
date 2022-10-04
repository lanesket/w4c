from turtle import forward
import pytorch_lightning as pl
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from meteopress.models.unet_attention.model import UNet_Attention
from dutils.evaluate import *


class ModelBase(pl.LightningModule):
    def __init__(self, params: dict) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.params = params
        self.loss = params['train']['loss']

        losses = {
            'MSE': F.mse_loss,
            'SmoothL1Loss': nn.SmoothL1Loss(),
            'L1': nn.L1Loss(),
            'BCELoss': nn.BCELoss(),
            'CrossEntropy': nn.CrossEntropyLoss(),
        }

        self.loss_fn = losses[self.loss]

    def retrieve_only_valid_pixels(self, x, mask):
        """ we asume 1s in mask are invalid pixels """
        return x[~mask]

    def get_target_mask(self, metadata):
        return metadata['target']['mask']

    def calculate_loss(self, pred, target, mask=None):
        if mask is not None:
            pred = self.retrieve_only_valid_pixels(pred, mask)
            target = self.retrieve_only_valid_pixels(target, mask)

        return self.loss_fn(pred, target)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=float(self.params['train']["lr"]),
                                      weight_decay=float(self.params['train']["weight_decay"]))
        return optimizer


class SmaAt(ModelBase):
    def __init__(self, params: dict):
        super().__init__(params)

        n_channels = self.params['dataset']['in_channels'] * \
            self.params['dataset']['len_seq_in']
        n_classes = self.params['dataset']['out_channels'] * \
            self.params['dataset']['len_seq_predict']

        self.model = UNet_Attention(
            n_channels=n_channels,
            n_classes=n_classes
        )

    def forward(self, x):
        x = x.flatten(1, 2)
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, meta = batch
        x = x.swapaxes(1, 2)
        pred = self(x)
        pred = pred.unsqueeze(1)

        mask = self.get_target_mask(meta)
        loss = self.calculate_loss(pred, y, mask)

        self.log('train_loss', loss,
                 batch_size=self.params['train']['batch_size'], sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, meta = batch
        x = x.swapaxes(1, 2)
        pred = self(x)
        pred = pred.unsqueeze(1)  # [B, 1, 32, 252, 252]

        mask = self.get_target_mask(meta)
        loss = self.calculate_loss(pred, y, mask)

        recall, precision, f1, acc, csi = recall_precision_f1_acc(
            y, pred.int())
        iou = iou_class(pred, y)

        self.log_dict(
            {
                'val_loss': loss,
                'val_recall': recall,
                'val_precision': precision,
                'val_f1': f1,
                'val_acc': acc,
                'val_csi': csi,
                'val_iou': iou
            },
            batch_size=self.params['train']['batch_size'], sync_dist=True)

        return loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x for x in outputs]).mean()
        self.log(f'val_loss_epoch', avg_loss, prog_bar=True,
                 batch_size=self.params['train']['batch_size'], sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, y, meta = batch
        x = x.swapaxes(1, 2)
        pred = self(x)
        pred = pred.unsqueeze(1)

        mask = self.get_target_mask(meta)
        loss = self.calculate_loss(pred, y, mask)

        recall, precision, f1, acc, csi = recall_precision_f1_acc(y, pred)
        iou = iou_class(pred, y)

        self.log_dict(
            {
                'test_loss': loss,
                'test_recall': recall,
                'test_precision': precision,
                'test_f1': f1,
                'test_acc': acc,
                'test_csi': csi,
                'test_iou': iou
            },
            batch_size=self.params['train']['batch_size'], sync_dist=True)
