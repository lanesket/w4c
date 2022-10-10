import argparse
from models.smaat import ModelBase, SmaAt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import datetime
import os
import torch

from models.unet_lightning import UNet_Lightning as UNetModel
from dutils.data_utils import load_config
from dutils.data_utils import get_cuda_memory_usage
from dutils.data_utils import tensor_to_submission_file
from dutils.w4c_dataloader import RainData
import wandb


class DataModule(pl.LightningDataModule):
    def __init__(self, params: dict, training_params: dict, mode: str):
        super().__init__()
        self.params = params
        self.training_params = training_params
        if mode in ['train']:
            self.train_ds = RainData('training', **self.params)
            self.val_ds = RainData('validation', **self.params)
        if mode in ['val']:
            self.val_ds = RainData('validation', **self.params)
        if mode in ['predict']:
            self.test_ds = RainData('test', **self.params)

    def __load_dataloader(self, dataset, shuffle=True, pin=True):
        dl = DataLoader(dataset,
                        batch_size=self.training_params['batch_size'],
                        num_workers=self.training_params['n_workers'],
                        shuffle=shuffle, pin_memory=pin, prefetch_factor=2,
                        persistent_workers=False)
        return dl

    def train_dataloader(self) -> DataLoader:
        return self.__load_dataloader(self.train_ds, shuffle=True, pin=True)

    def val_dataloader(self) -> DataLoader:
        return self.__load_dataloader(self.val_ds, shuffle=False, pin=True)

    def test_dataloader(self) -> DataLoader:
        return self.__load_dataloader(self.test_ds, shuffle=False, pin=True)


def get_trainer(gpus: list, params: dict) -> pl.Trainer:
    wandb.login(key="fc80d2f1bb37c383bfd7a9bb85cf0aac6d792d19", relogin=True)
    wandb_logger = pl.loggers.WandbLogger(
        name=params['experiment']['name'],
        project="weather4cast",
        entity='ctu-meteopress',
        save_code=True,
        # notes=params['experiment']['notes']
        )

    callbacks = [
        ModelCheckpoint(monitor='val_loss_epoch', save_top_k=3, save_last=True,
                        filename='{epoch:02d}-{val_loss_epoch:.6f}')
    ]

    if params['train']['early_stopping']:
        callback_funcs = callbacks.append(EarlyStopping(monitor="val_loss",
                                                        patience=params['train']['patience'],
                                                        mode="min"))

    return pl.Trainer(devices=gpus,
                      max_epochs=params['train']['max_epochs'],
                      gradient_clip_val=params['model']['gradient_clip_val'],
                      gradient_clip_algorithm=params['model']['gradient_clip_algorithm'],
                      accelerator="gpu",
                      callbacks=callback_funcs,
                      logger=wandb_logger,
                      profiler='simple',
                      precision=params['experiment']['precision'])


def get_model(params) -> ModelBase:
    return SmaAt(params)


def train(params: dict, gpus: list, mode: str):
    data = DataModule(params['dataset'], params['train'], mode)
    model = get_model(params)
    trainer = get_trainer(gpus, params)

    if mode == 'train':
        trainer.fit(model, data)
        wandb.finish()
    elif mode == 'predict':
        if len(params["dataset"]["regions"]) > 1 or params["predict"]["region_to_predict"] != str(params["dataset"]["regions"][0]):
            print("EXITING... \"regions\" and \"regions to predict\" must indicate the same region name in your config file.")
        else:
            scores = trainer.predict(model, data.test_dataloader())
            scores = torch.concat(scores)
            tensor_to_submission_file(scores, params['predict'])
            wandb.finish()


def set_parser():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-f", "--config_path", type=str, required=False, default='./configurations/config_basline.yaml',
                        help="path to config-yaml")
    parser.add_argument("-g", "--gpus", type=int, nargs='+', required=False, default=1,
                        help="specify gpu(s): 1 or 1 5 or 0 1 2 (-1 for no gpu)")
    parser.add_argument("-m", "--mode", type=str, required=False, default='train',
                        help="choose mode: train (default) / val / predict")
    parser.add_argument("-c", "--checkpoint", type=str, required=False, default='',
                        help="init a model from a checkpoint path. '' as default (random weights)")
    parser.add_argument("-n", "--name", type=str, required=False, default='',
                        help="Set the name of the experiment")
    parser.add_argument("-no", "--notes", type=str, required=False, default='',
                        help="Set the description of the experiment")

    return parser


def update_params_based_on_args(options):
    config_p = os.path.join('models/configurations', options.config_path)
    params = load_config(config_p)

    if options.name != '':
        print(params['experiment']['name'])
        params['experiment']['name'] = options.name
        params['experiment']['notes'] = options.notes + ''
    return params


if __name__ == "__main__":
    parser = set_parser()
    options = parser.parse_args()
    params = update_params_based_on_args(options)

    train(params, options.gpus, options.mode)
