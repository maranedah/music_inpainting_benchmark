
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

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='')

parser.add_argument('--model', type=str, default="SKETCHNET")
parser.add_argument('--mode', type=str, default="")
parser.add_argument('--vae_model', type=str, default="vae")
parser.add_argument('--dataset', type=str, default="folk")
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--repeat', type=int, default=1)
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

args = parser.parse_args()

PROJECT_DIR = Path(__file__).resolve().parents[0]
RAW_DIR = os.path.join(PROJECT_DIR, "data", "raw", args.dataset)
EXTERNAL_DIR = os.path.join(PROJECT_DIR, "data", "external", args.dataset)


def get_loaders(train, val, test):
    train_config = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": 16,
        "pin_memory": True,
        "drop_last": True,
    }
    val_config = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": 16,
        "pin_memory": True,
        "drop_last": True,
    }
    return [DataLoader(dataset, **train_config) if i==0 else DataLoader(dataset, **val_config) for i, dataset in enumerate((train, val, test))]


def get_datasets(df, model=args.model):
    return [get_dataset(model, args.dataset, df, set_, RAW_DIR, args.repeat, args.is_polyphony) for set_ in ["train", "val", "test"]]


df = get_clean_df(args.dataset)

if args.model in ["ARNN", "VLI", "DEEPBACH"]:
    train_set, val_set, test_set = get_datasets(df)
    train_loader, val_loader, test_loader = get_loaders(train_set, val_set, test_set)

    train_model = importlib.import_module(f"src.models.{args.model.lower()}.train_model")
    init_model = train_model.init_model
    epoch_step = train_model.epoch_step
    
    with torch.cuda.device(args.gpu_id):
        model, optimizer, scheduler = init_model(args)
        if args.train:
            train(
                model, 
                epoch_step, 
                args.model+args.mode, 
                args.dataset, 
                args.n_epochs, 
                train_loader, 
                val_loader, 
                None, 
                optimizer, 
                scheduler, 
                args
            )
            
        if args.generate:
            model.eval()
            metrics = Metrics()

            for i in range(len(test_set)):
                data = test_set[i][0]
                y_pred, y_true, y_past, y_future = generate_data(model, data, model_name=args.model)
                metrics.calc_metrics(y_true, y_pred, y_past, y_future)
                if args.model == "VLI":
                    noteseq_past = remi2noteseq(y_past, start_bar=0, bar_length=6, measure_size=16)
                    noteseq_future = remi2noteseq(y_future, start_bar=10, bar_length=6, measure_size=16)
                    noteseq_pred = remi2noteseq(y_pred, start_bar=6, bar_length=4, measure_size=16)
                    noteseq_true = remi2noteseq(y_true, start_bar=6, bar_length=4, measure_size=16)
                    noteseq_to_midi(noteseq_past + noteseq_true + noteseq_future, output=f"data/generated/{args.model}/{args.dataset}_{i}_true.mid")
                    noteseq_to_midi(noteseq_past + noteseq_pred + noteseq_future, output=f"data/generated/{args.model}/{args.dataset}_{i}_pred.mid")
                else:
                    noteseq_to_midi(list(y_past) + list(y_true) + list(y_future), output=f"data/generated/{args.model}/{args.dataset}_{i}_true.mid")
                    noteseq_to_midi(list(y_past) + list(y_pred) + list(y_future), output=f"data/generated/{args.model}/{args.dataset}_{i}_pred.mid")

                #breakpoint()
                #data2midi(y_true, output=f"y_true_{i}.mid")
                #data2midi(y_pred, output=f"y_pred_{i}.mid")
                
                print(f"{i}/{len(test_set)}") if i % 10 == 0 else None
                print(metrics.get_pos_f1(), metrics.get_px_acc(), metrics.get_rx_acc(), metrics.silence_divergence, metrics.px_similarity_div, metrics.groove_similarity_div)
                

if args.model in ["INPAINTNET", "SKETCHNET", "GRU_VAE", "VAE_ATTENTION"]:

    vae_sets = get_datasets(df, model=f"vae_{args.model}")
    vae_train_set, vae_val_set, vae_test_set = vae_sets
    vae_train_loader, vae_val_loader, vae_test_loader = get_loaders(vae_train_set, vae_val_set, vae_test_set)

    model_sets = get_datasets(df, model=args.model)
    train_set, val_set, test_set = model_sets
    train_loader, val_loader, test_loader = get_loaders(train_set, val_set, test_set)
    
    train_model = importlib.import_module(f"src.models.{args.model.lower()}.train_model")
    init_vae = train_model.init_vae
    init_model = train_model.init_model
    epoch_step = train_model.epoch_step
    vae_epoch_step = train_model.vae_epoch_step
    
    
    with torch.cuda.device(args.gpu_id):

        vae_model, vae_optimizer, vae_scheduler = init_vae(args)
        if args.train:
            if args.train_vae:
                train(
                    vae_model, 
                    vae_epoch_step, 
                    f"vae_{args.model}", 
                    args.dataset, 
                    args.vae_n_epochs, 
                    vae_train_loader, 
                    vae_val_loader, 
                    vae_loss_function, 
                    vae_optimizer, 
                    vae_scheduler, 
                    args
                )
            print("sali de train_vae")
            model, optimizer, scheduler = init_model(vae_model, args)
            train(
                model, 
                epoch_step, 
                args.model, 
                args.dataset, 
                args.n_epochs, 
                train_loader, 
                val_loader, 
                F.cross_entropy, 
                optimizer, 
                scheduler, 
                args
            )

        if args.generate:
            model, optimizer, scheduler = init_model(vae_model, args)
            model.eval()
            metrics = Metrics()

            for i in range(len(test_set)):
                data = test_set[i][0]
                y_pred, y_true, y_past, y_future = generate_data(model, data, model_name=args.model)
                metrics.calc_metrics(y_true, y_pred, y_past, y_future)
                noteseq_to_midi(list(y_past) + list(y_true) + list(y_future), output=f"data/generated/{args.model}/{args.dataset}_{i}_true.mid")
                noteseq_to_midi(list(y_past) + list(y_pred) + list(y_future), output=f"data/generated/{args.model}/{args.dataset}_{i}_pred.mid")

                #data2midi(y_true, output=f"y_true_{i}.mid")
                #data2midi(y_pred, output=f"y_pred_{i}.mid")
                
                print(f"{i}/{len(test_set)}") if i % 10 == 0 else None
                print(metrics.get_pos_f1(), metrics.get_px_acc(), metrics.get_rx_acc(), metrics.silence_divergence, metrics.px_similarity_div, metrics.groove_similarity_div)
