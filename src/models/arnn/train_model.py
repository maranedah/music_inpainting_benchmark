
import torch
import os
import numpy as np
import pandas as pd
import sys
import time

from src.report.logs import print_and_save_logs
from src.report.metrics import Metrics

from torch import optim
from torch.nn import functional as F
from torch import nn, distributions
#from torch.optim.lr_scheduler import ExponentialLR, CyclicLR

from pathlib import Path
import os

from .anticipation_rnn import AnticipationRNN

PROJECT_DIR = Path(__file__).resolve().parents[3]
MODELS_DIR = os.path.join(PROJECT_DIR, "models")


def init_model(args):
    model = AnticipationRNN()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    
    model_path = os.path.join(MODELS_DIR, f"{args.model}_{args.dataset}.pt")
    
    if torch.cuda.is_available():
        print('Using: ', torch.cuda.get_device_name(torch.cuda.current_device()))
        model.cuda(torch.cuda.current_device())
    else:
        raise Exception("You are not using cuda GPU") 
    
    if os.path.isfile(model_path):
        checkpoint = torch.load(model_path)

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        #scheduler.load_state_dict(checkpoint["scheduler"])

    
    return model, optimizer, None 


def epoch_step(model, data, loss_func, optimizer, scheduler):
    device = torch.cuda.current_device()
    metrics = Metrics()
    start_time = time.time()

    subdivision = 4
    num_voices = 1
        
    for i, (d, label) in enumerate(data):
        
        # param metadata: (batch_size, num_voices, chorale_length, num_metadatas)
        # metadata es la misma shape que data, pero con una dimension mas que contiene [subdivision, num_voices]
        d = d.long().to(device = device, non_blocking = True)
        
        if model.training:
            optimizer.zero_grad()
        
        metadata_tensor = torch.zeros(d.shape[0], d.shape[1], d.shape[2], 2)
        metadata_content = torch.Tensor([subdivision,num_voices]) # [subdivision, num_voices]
        metadata = metadata_tensor + metadata_content
        metadata = metadata.long().to(device = device, non_blocking = True)

        constraints_loc = torch.cat((torch.ones([64, 1, 24*6]), torch.zeros([64, 1, 24*4]), torch.ones([64, 1, 24*6])), dim=-1).long()
        weights = model.forward(
            chorale=d, 
            metadata=torch.zeros([64, 1, 384, 2]).long().cuda(),
            constraints_loc=constraints_loc
            )
        
        weights = weights[:, :, 6*24:10*24, :]
        targets = d.transpose(0,1)[:, :, 6*24:10*24]
        
        # list of (batch, num_notes)
        loss = F.cross_entropy(weights.reshape(-1, weights.size(-1)), targets.reshape(-1))
        
        y_pred = weights.reshape(-1, weights.size(-1)).max(-1)[-1]
        y_true = targets.reshape(-1)

        if model.training:
            loss.backward()
            optimizer.step()
            if scheduler != None:
                scheduler.step()

        metrics.loss += loss.detach().item()
        y_true = y_true.reshape(64, -1)
        y_pred = y_pred.reshape(64, -1)
        #for i in range(64):
        metrics.calc_metrics(y_true[0].detach().to('cpu').numpy(), y_pred[0].detach().to('cpu').numpy())

        print_and_save_logs(metrics, model.training, i, len(data), start_time, save=False)
    
    return metrics

def generate(model, d):
    device = torch.cuda.current_device()
    subdivision = 4
    num_voices = 1
    
    
    d = torch.from_numpy(d).unsqueeze(0).long().to(device = device, non_blocking = True)
    metadata_tensor = torch.zeros(d.shape[0], d.shape[1], 2)
    metadata_content = torch.Tensor([subdivision,num_voices]) # [subdivision, num_voices]
    metadata = metadata_tensor + metadata_content
    metadata = metadata.long().to(device = device, non_blocking = True)

    constraints_loc = torch.cat((torch.ones([1, 1, 24*6]), torch.zeros([1, 1, 24*4]), torch.ones([1, 1, 24*6])), dim=-1).long()
    weights = model.forward(
        chorale=d, 
        metadata=torch.zeros([1, 1, 384, 2]).long().cuda(),
        constraints_loc=constraints_loc
        )
    y_pred = weights.reshape(-1, weights.size(-1)).max(-1)[-1]
    return y_pred[24*6:24*10]    
    
    