import torch
import argparse
import os
import time
from pathlib import Path

from src.report.logs import print_and_save_logs
from src.report.metrics import Metrics


PROJECT_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = os.path.join(PROJECT_DIR, "models")

def train(model, epoch_step_func, model_name, dataset_name, n_epochs, train_data, val_data, loss_func, optimizer, scheduler, args):
        
    best_val_loss = float('inf')
    start_time = time.time()
    patience = args.patience
    counter = 0

    for epoch in range(args.n_epochs):
        print(f"Epoch:{epoch}")
        # Entrenar
        print(f"Training {args.model}...")
        model.train()
        train_metrics  = epoch_step_func(model, train_data, loss_func, optimizer, scheduler)
        print(f"Validating {args.model}...")
        # Evaluar (val = validacion)
        with torch.no_grad():
            model.eval()
            val_metrics = epoch_step_func(model, val_data, loss_func, optimizer, scheduler)

        print_and_save_logs(train_metrics, True, epoch, n_epochs, start_time, model_name, args.dataset, keep_print=True)
        print_and_save_logs(val_metrics, False, epoch, n_epochs, start_time, model_name, args.dataset, keep_print=True)

        print("new metric", val_metrics.get_loss())
        print("best val loss", best_val_loss)
        if val_metrics.get_loss() < best_val_loss:
            counter = 0 
            best_val_loss = val_metrics.get_loss()
            checkpoint = {
                "model": model.cpu().state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler else None
            }
            torch.save(checkpoint, os.path.join(MODELS_DIR, f"{model_name}_{dataset_name}.pt"))  
            model.cuda() 
        else:
            counter += 1
            if counter >= patience:
                break