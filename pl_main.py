import argparse
import os

import optuna
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.models as models
from pytorch_lightning.loggers import MLFlowLogger

import torch
import argparse
import os
import importlib

from torch.utils.data import DataLoader
from torch.nn import functional as F
from pathlib import Path
import pretty_midi as pyd

from src.data.clean_data import get_clean_df
from src.models.trainer import train
from src.models.sketchnet.utils import vae_loss_function
from src.generative.data_to_midi import noteseq_to_midi
from src.generative.generate_data import generate_data
from src.report.metrics import Metrics
from src.report.metrics import remi2noteseq
from src.features.midi_dataset import get_dataset

from src.models.arnn.anticipation_rnn import LitAnticipationRNN
from src.models.gru_vae.gru_vae import LitGRU_VAE

import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description="")

parser.add_argument('--model', type=str, default="SKETCHNET")
parser.add_argument('--mode', type=str, default="")
parser.add_argument('--vae_model', type=str, default="vae")
parser.add_argument('--dataset', type=str, default="folk")
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--repeat', type=int, default=10)
parser.add_argument('--is_polyphony', type=bool, default=False)

parser.add_argument('--patience', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--n_epochs', type=int, default=1000)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--scheduler', type=str, default=None)

parser.add_argument('--train_vae')
parser.add_argument('--vae_batch_size', type=int, default=32)
parser.add_argument('--vae_n_epochs', type=int, default=200)
parser.add_argument('--vae_lr', type=float, default=1e-4)
parser.add_argument('--vae_decay', type=float, default=0.9999)
parser.add_argument('--vae_min_decay', type=float, default=1e-5)

#INPAINTNET
parser.add_argument('--num_latent_rnn_layers', type=int, default=2)
parser.add_argument('--latent_rnn_hidden_size', type=int, default=512)
parser.add_argument('--latent_rnn_dropout_prob', type=float, default=0.5)
parser.add_argument('--auto_reg', type=bool, default=True)
parser.add_argument('--teacher_forcing', type=bool, default=True)

parser.add_argument('--train')
parser.add_argument('--generate')

# Scheduler params
parser.add_argument("--sch_factor", type=float, default=5)
parser.add_argument("--sch_step", type=int, default=100)
parser.add_argument("--sch_mode", type=str, default="exp_range")
parser.add_argument("--sch_gamma", type=float, default=1 - 1e-4)
parser.add_argument("--sch_momentum", type=bool, default=False)


args = parser.parse_args()

PROJECT_DIR = Path(__file__).resolve().parents[0]
RAW_DIR = os.path.join(PROJECT_DIR, "data", "raw", args.dataset)
EXTERNAL_DIR = os.path.join(PROJECT_DIR, "data", "external", args.dataset)

def get_loaders(train, val, test):
    train_config = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": 8,
        "pin_memory": True,
        "drop_last": True,
    }
    val_config = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": 8,
        "pin_memory": True,
        "drop_last": True,
    }
    return [DataLoader(dataset, **train_config) if i==0 else DataLoader(dataset, **val_config) for i, dataset in enumerate((train, val, test))]

def get_datasets(df, model=args.model):
    return [get_dataset(model, args.dataset, df, set_, RAW_DIR, args.repeat, args.is_polyphony) for set_ in ["train", "val", "test"]]


def objective(trial):

    model_name = args.model

    if model_name == "ARNN":
        note_embedding_dim=20,
        metadata_embedding_dim=2, #30
        num_lstm_constraints_units=256,
        num_lstm_generation_units=256,
        linear_hidden_size=256, #128
        num_layers=2, #1
        dropout_input_prob=0.2,
        dropout_prob=0.2, #0.5
        unary_constraint=True,
        no_metadata=False
        learn_rate = trial.suggest_float("learn_rate", 1e-5, 1e-3)
        batch_size = 8
                 

        lit_model = LitAnticipationRNN(
            note_embedding_dim=note_embedding_dim,
            metadata_embedding_dim=metadata_embedding_dim, #30
            num_lstm_constraints_units=num_lstm_constraints_units,
            num_lstm_generation_units=num_lstm_generation_units,
            linear_hidden_size=linear_hidden_size, #128
            num_layers=num_layers, #1
            dropout_input_prob=dropout_input_prob,
            dropout_prob=dropout_prob, #0.5
            unary_constraint=unary_constraint,
            no_metadata=no_metadata,
            learn_rate=learn_rate,
            batch_size=batch_size,
            sch_factor=args.sch_factor,
            sch_step=args.sch_step,
            sch_mode=args.sch_mode,
            sch_gamma=args.sch_gamma,
            sch_momentum=args.sch_momentum,
        )
    if model_name == "GRU_VAE":

        zp_dims = 128
        zr_dims = 128
        pf_num = 2
        inpaint_len = 4
        gen_dims = 256
        learn_rate = trial.suggest_float("learn_rate", 1e-5, 1e-3)
        
        
        train_model = importlib.import_module(f"src.models.{args.model.lower()}.train_model")
        init_vae = train_model.init_vae
        vae_model, vae_optimizer, vae_scheduler = init_vae(args)
        for param in vae_model.parameters():
            param.requires_grad = False
        
        lit_model = LitGRU_VAE(
            zp_dims = zp_dims,
            zr_dims = zr_dims,
            pf_num = pf_num,
            inpaint_len = inpaint_len,
            gen_dims = gen_dims,
            vae_model = vae_model.to("cuda:0"),
            learn_rate=learn_rate,
            batch_size=args.batch_size,
            sch_factor=args.sch_factor,
            sch_step=args.sch_step,
            sch_mode=args.sch_mode,
            sch_gamma=args.sch_gamma,
            sch_momentum=args.sch_momentum,
        )


    df = get_clean_df(args.dataset)
    if args.model in ["ARNN", "VLI", "DEEPBACH"]:
        train_set, val_set, test_set = get_datasets(df)
        train_loader, val_loader, test_loader = get_loaders(train_set, val_set, test_set)
    else:
        vae_sets = get_datasets(df, model=f"vae_{args.model}")
        vae_train_set, vae_val_set, vae_test_set = vae_sets
        vae_train_loader, vae_val_loader, vae_test_loader = get_loaders(vae_train_set, vae_val_set, vae_test_set)

        model_sets = get_datasets(df, model=args.model)
        train_set, val_set, test_set = model_sets
        train_loader, val_loader, test_loader = get_loaders(train_set, val_set, test_set)

    mlf_logger = MLFlowLogger(
        experiment_name=f"{args.model}_data", tracking_uri="file:./mlruns"
    )
    trainer = pl.Trainer(
        callbacks=[
            pl.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=args.patience,
                mode="min",
            ),
            pl.callbacks.ModelCheckpoint(
                monitor="val_loss",
                filename="best_checkpoint",
                save_top_k=0,
                mode="min",
            ),
        ],
        precision=16,
        deterministic=False,
        logger=mlf_logger,
        gpus=[args.gpu_id],
        #min_epochs=args.min_epochs,
    )

    trainer.fit(
        model=lit_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
    return trainer.callback_metrics["val_loss"].item()


def train(n_trials):
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(
        objective,
        n_trials=n_trials,
    )

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


if __name__ == "__main__":
    train(100)
