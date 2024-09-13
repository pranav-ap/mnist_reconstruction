from utils.logger_setup import logger
from dataclasses import dataclass

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import lightning as L
import lightning.pytorch as pl

torch.set_float32_matmul_precision('medium')


"""
Configuration
"""

@dataclass
class TrainingConfig:
    train_batch_size = 128
    val_batch_size = 128

    max_epochs = 20
    check_val_every_n_epoch = 2
    log_every_n_steps = 50
    accumulate_grad_batches = 1
    learning_rate = 1e-3 # 4

    data_dir = "D:/mnist_reconstruction/data"
    output_dir = "D:/mnist_reconstruction/data"
    # data_dir = "/kaggle/working"
    # output_dir = "/kaggle/working"


config = TrainingConfig()



"""
Dataset Classes
"""


class MNISTNoisyDataset(torchvision.datasets.MNIST):
    def __init__(self, *args, noisy_transform, transform=None, **kwargs):
        # No transform at initialization
        super().__init__(*args, transform=None, **kwargs)

        # Store the transformation (will be applied later)
        self.normal_transform = transform
        self.noisy_transform = noisy_transform

    def __getitem__(self, index):
        original_image, target = super().__getitem__(index)

        normal_image = self.normal_transform(original_image)
        noisy_image = self.noisy_transform(original_image)

        return normal_image.float(), noisy_image.float(), target


class MNISTNoisyReconDataModule(L.LightningDataModule):
    def __init__(self):
        super().__init__()

        self.num_workers = os.cpu_count()  # <- use all available CPU cores
        num_gpus = torch.cuda.device_count()
        if num_gpus > 0:
            self.num_workers = 2 * num_gpus

        self.normal_transform = T.Compose([
            T.ToTensor(),
#             T.Normalize((0.5,), (0.5,)),
        ])

        from noise_transform import noise_transform
        self.noise_transform = noise_transform

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str):
        if stage == "fit" or stage == "validate":
            full_train_dataset = MNISTNoisyDataset(
                root=f'{config.data_dir}/train',
                train=True,
                download=True,
                transform=self.normal_transform,
                noisy_transform=self.noise_transform
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
            self.test_dataset = MNISTNoisyDataset(
                root=f'{config.data_dir}/test',
                train=False,
                download=True,
                transform=self.normal_transform,
                noisy_transform=self.noise_transform
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


class MNISTNoisyReconAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.enc3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.enc4 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2,2)

        self.dec1 = nn.ConvTranspose2d(8, 8, kernel_size=3, stride=2)
        self.dec2 = nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2)
        self.dec3 = nn.ConvTranspose2d(16, 32, kernel_size=2, stride=2)
        self.dec4 = nn.ConvTranspose2d(32, 64, kernel_size=2, stride=2)
        self.out = nn.Conv2d(64, 1, kernel_size=3, padding=1)


    def forward(self, x):
        # Encoder
        x = self.pool(F.relu(self.enc1(x)))
        x = self.pool(F.relu(self.enc2(x)))
        x = self.pool(F.relu(self.enc3(x)))
        x = self.pool(F.relu(self.enc4(x))) # latent space representation

        # Decoder
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        x = F.relu(self.dec4(x))
        x = torch.sigmoid(self.out(x))
        return x


"""
Lightning Module
"""


class MNISTNoisyReconLightning(pl.LightningModule):
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
        X, X_noisy, _ = batch
        X_reconstructed = self.model(X_noisy)
        loss = F.mse_loss(X_reconstructed, X)
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
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
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
            patience=3,
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

    autoencoder_model = MNISTNoisyReconAutoencoder()
    lightning_module = MNISTNoisyReconLightning(autoencoder_model, flat_image_length)
    dm = MNISTNoisyReconDataModule()

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

    #     fast_dev_run=True,
    #     overfit_batches=1,
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



