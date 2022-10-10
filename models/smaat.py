import pytorch_lightning as pl
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from meteopress.models.unet_attention.model import UNet_Attention
from dutils.evaluate import *
from torchvision.utils import make_grid
from models.losses import *


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
            'BCEWithLogitsLoss': nn.BCEWithLogitsLoss(pos_weight=torch.tensor((params['train']['pos_weight']))),
            'CrossEntropy': nn.CrossEntropyLoss(),
            'DiceLoss': DiceLoss(),
            'FocalLoss': FocalLoss(),
            'DiceFocalLoss': DiceFocalLoss(),
            'DiceBCELoss': DiceBCELoss()
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

        if (self.loss in ['CrossEntropy', 'FocalLoss']):
            pred = pred.view((1, -1))
            target = target.view((1, -1))

        return self.loss_fn(pred, target)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(self.params['train']["lr"]),
            weight_decay=float(self.params['train']["weight_decay"]))

        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                    mode="min",
                                                                    factor=0.1,
                                                                    patience=4),
            'monitor': 'val_loss_epoch',
        }

        return [optimizer], [scheduler]


class SRCNN(nn.Module):
    def __init__(self, channels):
        super(SRCNN, self).__init__()

        self.model = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(channels, channels * 2, kernel_size=9, padding=2,
                      padding_mode='replicate'),
            nn.ReLU(inplace=True),

            nn.Conv2d(channels * 2, channels, kernel_size=1, padding=2,
                      padding_mode='replicate'),

            nn.Upsample(scale_factor=3, mode='bilinear', align_corners=True),
            torch.nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=5, padding=2,
                      padding_mode='replicate'),
            torch.nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, padding=0,
                      padding_mode='replicate'),
        )

    def forward(self, x):
        return self.model(x)


class SmaAt(ModelBase):
    def __init__(self, params: dict):
        super().__init__(params)

        n_channels = self.params['dataset']['in_channels'] * \
            self.params['dataset']['len_seq_in']
        n_classes = self.params['dataset']['out_channels'] * \
            self.params['dataset']['len_seq_predict']

        self.model = UNet_Attention(
            n_channels=n_channels,
            n_classes=n_classes,
            dropout=self.params['train']['dropout_rate']
        )

        self.srcnn = SRCNN(n_classes)

        # satellite - 12x12 km region
        # radar - 2x2 km region
        image_size = 252
        crop_size = int((2 / 12) * image_size)
        self.radar_crop = T.CenterCrop((crop_size, crop_size))

    def forward(self, x):
        x = x.flatten(1, 2)
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, meta = batch
        x = x.swapaxes(1, 2)
        pred = self(x)
        pred = self.radar_crop(pred)
        # 32 -> 252
        pred = self.srcnn(pred)
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
        pred = self.radar_crop(pred)
        # 32 -> 252
        pred = self.srcnn(pred)
        pred = pred.unsqueeze(1)  # [B, 1, 32, 252, 252]

        mask = self.get_target_mask(meta)
        loss = self.calculate_loss(pred, y, mask)

        idx_gt = pred > 0
        pred[idx_gt] = 1
        pred[~idx_gt] = 0

        wandb.log({
            'Precipitation Map': [wandb.Image(make_grid(pred[0].swapaxes(0, 1)), caption='Prediction'),
                                  wandb.Image(make_grid(y[0].swapaxes(0, 1)), caption='GT')]
        })

        recall, precision, f1, acc, csi = recall_precision_f1_acc(y, pred)
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
        pred = self.radar_crop(pred)
        # 32 -> 252
        pred = self.srcnn(pred)
        pred = pred.unsqueeze(1)

        mask = self.get_target_mask(meta)
        loss = self.calculate_loss(pred, y, mask)

        idx_gt0 = pred >= 0
        pred[idx_gt0] = 1
        pred[~idx_gt0] = 0

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

    def predict_step(self, batch, batch_idx):
        x, _, _ = batch
        x = x.swapaxes(1, 2)
        pred = self(x)
        pred = self.radar_crop(pred)
        # 32 -> 252
        pred = self.srcnn(pred)
        pred = pred.unsqueeze(1)

        idx_gt0 = pred >= 0
        pred[idx_gt0] = 1
        pred[~idx_gt0] = 0

        return pred
