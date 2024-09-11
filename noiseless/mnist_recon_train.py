from utils.logger_setup import logger
from dataclasses import dataclass

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import lightning as L
import lightning.pytorch as pl

torch.set_float32_matmul_precision('medium')


"""
Configuration
"""


@dataclass
class TrainingConfig:
    train_batch_size = 512
    val_batch_size = 512

    max_epochs = 100
    check_val_every_n_epoch = 5
    log_every_n_steps = 100
    accumulate_grad_batches = 1
    learning_rate = 1e-4

    data_dir = "D:/mnist_reconstruction/data"
    output_dir = "D:/mnist_reconstruction/data"
    # data_dir = "/kaggle/working"
    # output_dir = "/kaggle/working"


config = TrainingConfig()


"""
Dataset Classes
"""


class MNISTReconDataModule(L.LightningDataModule):
    def __init__(self):
        super().__init__()

        self.num_workers = os.cpu_count()  # <- use all available CPU cores
        num_gpus = torch.cuda.device_count()
        if num_gpus > 0:
            self.num_workers = 2 * num_gpus

        from torchvision import transforms

        self.transform = transforms.Compose([
            # Convert image to tensor (scales pixel values to [0, 1])
            transforms.ToTensor(),
            # Normalize to have mean 0.5 and std 0.5
            # transforms.Normalize((0.5,), (0.5,)),
        ])

        self.reverse_transform = transforms.Compose([
            # transforms.Normalize(mean=[-0.5 / 0.5], std=[1 / 0.5]),  # Reverse normalization
            transforms.ToPILImage()  # Convert tensor to PIL image
        ])

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str):
        if stage == "fit" or stage == "validate":
            full_train_dataset = torchvision.datasets.MNIST(
                root=f'{config.data_dir}/train',
                train=True,
                transform=self.transform,
                download=True
            )

            # Define the size of the splits
            train_size = int(0.8 * len(full_train_dataset))
            val_size = len(full_train_dataset) - train_size

            # Perform the split
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                full_train_dataset,
                [train_size, val_size]
            )

            logger.info(f"Total Dataset       : {len(self.train_dataset) + len(self.val_dataset)} samples")
            logger.info(f"Train Dataset       : {len(self.train_dataset)} samples")
            logger.info(f"Validation Dataset  : {len(self.val_dataset)} samples")

        if stage == 'test':
            self.test_dataset = torchvision.datasets.MNIST(
                root=f'{config.data_dir}/test',
                train=False,
                transform=self.transform,
                download=True
            )

            logger.info(f"Test Dataset  : {len(self.test_dataset)} samples")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=config.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=config.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=config.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )


"""
Model Classes
"""


class MNISTReconAutoencoder(nn.Module):
    def __init__(self, flat_image_length, encoding_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(flat_image_length, 256),
            nn.Sigmoid(),
            nn.Linear(256, encoding_dim),
            nn.Sigmoid()
        )

        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, flat_image_length),
            nn.Sigmoid()
        )

    def forward(self, x):
#         logger.debug(f'Input shape : {x.shape}')

        # Flatten the image before passing to the model
        x = x.view(x.size(0), -1)
#         logger.debug(f'Flattened : {x.shape}')

        x = self.encoder(x)
#         logger.debug(f'Encoder Output shape : {x.shape}')

        x = self.decoder(x)
#         logger.debug(f'Decoder Output shape : {x.shape}')

        # Reshape back to the original image shape after decoding
        x = x.view(x.size(0), 1, 28, 28)
#         logger.debug(f'Reshape back to original : {x.shape}')

        return x


"""
Lightning Module
"""


class MNISTReconLightning(pl.LightningModule):
    def __init__(self, model, flat_image_length):
        super().__init__()

        self.model = model
        self.flat_image_length = flat_image_length

        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Number of Trainable Parameters : {total_trainable_params}")

        self.learning_rate = config.learning_rate

        self.save_hyperparameters(ignore=['model'])

    def forward(self, X):
        X_reconstructed = self.model(X)
        return X_reconstructed

    def shared_step(self, batch):
        X, _ = batch
        X_reconstructed = self.model(X)

        X_flat = torch.reshape(X, shape=(-1, self.flat_image_length))
        X_reconstructed_flat = torch.reshape(X_reconstructed, shape=(-1, self.flat_image_length))

        loss = F.mse_loss(X_reconstructed_flat, X_flat)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "train_loss",
            }
        }

    def configure_callbacks(self):
        early_stop = L.pytorch.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            min_delta=0.00,
            patience=4,
            verbose=False,
        )

        checkpoint = L.pytorch.callbacks.ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            dirpath=f'{config.output_dir}/checkpoints/',
            save_top_k=1,
            save_last=True
        )

        progress_bar = L.pytorch.callbacks.TQDMProgressBar(process_position=0)
        lr_monitor = L.pytorch.callbacks.LearningRateMonitor(logging_interval='step')
        # summary = ModelSummary(max_depth=-1)
        # swa = StochasticWeightAveraging(swa_lrs=1e-2)

        return [checkpoint, progress_bar, lr_monitor, early_stop]


"""
Train Function
"""


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    encoding_dim = 128
    flat_image_length = 28 * 28

    autoencoder_model = MNISTReconAutoencoder(flat_image_length, encoding_dim)
    lightning_module = MNISTReconLightning(autoencoder_model, flat_image_length)
    dm = MNISTReconDataModule()

    trainer = pl.Trainer(
        default_root_dir=f"{config.output_dir}/",
        logger=L.pytorch.loggers.CSVLogger(save_dir=f'{config.output_dir}/'),
        devices='auto',
        accelerator="auto",  # auto, gpu, cpu, ...

        max_epochs=config.max_epochs,
        log_every_n_steps=config.log_every_n_steps,
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        accumulate_grad_batches=config.accumulate_grad_batches,
        # gradient_clip_val=0.1,

        fast_dev_run=True,
        # overfit_batches=1,
        num_sanity_val_steps=1,
        enable_model_summary=False,
    )

    trainer.fit(
        lightning_module,
        datamodule=dm,
        # ckpt_path=f'{config.output_dir}\\checkpoints\\last.ckpt'
    )

    best_model_path = trainer.checkpoint_callback.best_model_path
    logger.info(f"Best model path : {best_model_path}")



