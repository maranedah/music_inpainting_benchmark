import os
from collections import namedtuple

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.loggers import MLFlowLogger

from src.features.dataset import Dataset
from src.models.vaes.sketchvae import LitSketchVAE
from src.models.vaes.sketchvae_v2 import LitSketchVAE_v2

file = open("hiperparameters.yaml")
hiperparameters = yaml.load(file, Loader=yaml.FullLoader)

batch_size = 512
model = "sketch_vae_v2"
dataset = "IrishFolkSong"
resolution = 24
hparams = namedtuple("ModelParams", hiperparameters[model].keys())(
    **hiperparameters[model]
)
optimizer = namedtuple("Optimizer", hparams.optimizer.keys())(**hparams.optimizer)
scheduler = namedtuple("Scheduler", hparams.scheduler.keys())(**hparams.scheduler)

train_set, val_set, test_set = [
    Dataset(
        dataset,
        hparams.data_type,
        split,
        hparams.format,
        resolution,
        hparams.data_transformations,
    )
    for split in ["train", "val", "test"]
]

train_loader, val_loader, test_loader = [
    torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=i == 0,
        num_workers=os.cpu_count(),
        pin_memory=True,
        drop_last=True,
    )
    for i, dataset in enumerate([train_set, val_set, test_set])
]


if model == "sketch_vae":
    lit_model = LitSketchVAE(
        input_dims=hparams.input_dims,
        p_input_dims=hparams.pitch_dims,
        r_input_dims=hparams.rhythm_dims,
        hidden_dims=hparams.hidden_dims,
        zp_dims=hparams.zp_dims,
        zr_dims=hparams.zr_dims,
        seq_len=hparams.seq_len,
        beat_num=hparams.beat_num,
        tick_num=hparams.tick_num,
        decay=hparams.decay,
        batch_size=batch_size,
        optimizer=optimizer,
        scheduler=scheduler,
    )

elif model == "sketch_vae_v2":
    lit_model = LitSketchVAE_v2(
        input_dims=hparams.input_dims,
        p_input_dims=hparams.pitch_dims,
        r_input_dims=hparams.rhythm_dims,
        hidden_dims=hparams.hidden_dims,
        zp_dims=hparams.zp_dims,
        zr_dims=hparams.zr_dims,
        seq_len=hparams.seq_len,
        beat_num=hparams.beat_num,
        tick_num=hparams.tick_num,
        decay=hparams.decay,
        batch_size=batch_size,
        optimizer=optimizer,
        scheduler=scheduler,
    )


experiment_name = f"{model}_{dataset}_{resolution}"
mlf_logger = MLFlowLogger(
    experiment_name=experiment_name,
    tracking_uri="file:./mlruns",
)

trainer = pl.Trainer(
    callbacks=[
        pl.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            mode="min",
        ),
        pl.callbacks.ModelCheckpoint(
            monitor="val_loss",
            filename="best_checkpoint",
            save_top_k=1,
            mode="min",
        ),
    ],
    max_epochs=-1,
    precision=16,
    deterministic=False,
    logger=mlf_logger,
    gpus=[0],
    min_epochs=10,
    gradient_clip_val=hparams.clip_grad,
)

trainer.fit(
    model=lit_model,
    train_dataloaders=train_loader,
    val_dataloaders=val_loader,
)
