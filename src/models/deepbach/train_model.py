
import torch
import os
import numpy as np
import pandas as pd
import sys
import time
import random

from src.report.logs import print_and_save_logs
from src.report.metrics import Metrics

import argparse

from torch import optim
from torch.nn import functional as F
from torch import nn, distributions

from pathlib import Path
import os

from .deepbach import DeepBach


PROJECT_DIR = Path(__file__).resolve().parents[3]
MODELS_DIR = os.path.join(PROJECT_DIR, "models")

class OptimizerGroup:
    def __init__(self, optimizers):
        self.optimizers = optimizers
    
    def state_dict(self):
        return [optimizer.state_dict() for optimizer in self.optimizers]

    def load_state_dict(self, dict_):
        return [self.optimizers[i].load_state_dict(dict_[i]) for i, optimizer in enumerate(self.optimizers)]


def init_deepbach_model(args):
    model = DeepBach(
        dataset=None,
        note_embedding_dim=20,
        meta_embedding_dim=20,
        num_layers=2,
        lstm_hidden_size=256,
        dropout_lstm=0.5,
        linear_hidden_size=256
    )
    
    #optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    #optimizer = [optim.Adam(voice_model.parameters(), lr=args.lr) for voice_model in model.voice_models]        
    optimizer = OptimizerGroup([optim.Adam(filter(lambda p: p.requires_grad, voice_model.parameters()), lr=args.lr) for voice_model in model.voice_models])        
    
    model_path = os.path.join(MODELS_DIR, f"{args.model}_{args.dataset}.pt")
    
    if torch.cuda.is_available():
        print('Using: ', torch.cuda.get_device_name(torch.cuda.current_device()))
        for voice_model in model.voice_models:
            voice_model.cuda(torch.cuda.current_device())
    else:
        raise Exception("You are not using cuda GPU") 
    
    if os.path.isfile(model_path):
        checkpoint = torch.load(model_path)

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        #scheduler.load_state_dict(checkpoint["scheduler"])

    return model, optimizer, None 
    

def preprocess_input(tensor_chorale, tensor_metadata, main_voice_index):
    batch_size, num_voices, chorale_length_ticks = tensor_chorale.size()

    # random shift! Depends on the dataset
    subdivision = 4
    offset = random.randint(0, subdivision)
    time_index_ticks = chorale_length_ticks // 2 + offset # TODO: CHECK THIS
    # split notes
    notes, label = preprocess_notes(tensor_chorale, time_index_ticks, main_voice_index)
    metas = preprocess_metas(tensor_metadata, time_index_ticks, main_voice_index)
    return notes, metas, label

def mask_entry(tensor, entry_index, dim):
    """
    Masks entry entry_index on dim dim
    similar to
    torch.cat((	tensor[ :entry_index],	tensor[ entry_index + 1 :], 0)
    but on another dimension
    :param tensor:
    :param entry_index:
    :param dim:
    :return:
    """
    mask = torch.ones_like(tensor)
    mask[:, entry_index] = 0
    mask = mask.to(torch.bool)
    tensor = torch.masked_select(tensor, mask).view(tensor.shape[0],-1)
    return tensor

def preprocess_notes(tensor_chorale, time_index_ticks, main_voice_index):
    batch_size, num_voices, _ = tensor_chorale.size()
    left_notes = tensor_chorale[:, :, :time_index_ticks]
    right_notes = torch.flip(tensor_chorale[:, :, time_index_ticks + 1:].unsqueeze(2), dims=[2,3]).squeeze(2)
    num_voices = 4
    if num_voices == 1:
        central_notes = None
    else:
        central_notes = mask_entry(tensor_chorale[:, :, time_index_ticks], entry_index=main_voice_index, dim=1)
    label = tensor_chorale[:, main_voice_index, time_index_ticks]
    return (left_notes, central_notes, right_notes), label


def preprocess_metas(tensor_metadata, time_index_ticks, main_voice_index):
    
    left_metas = tensor_metadata[:, main_voice_index, :time_index_ticks, :]
    right_metas = torch.flip(tensor_metadata[:, main_voice_index, time_index_ticks + 1:, :].unsqueeze(1), dims=[1,2]).squeeze(1)
    central_metas = tensor_metadata[:, main_voice_index, time_index_ticks, :]
    return left_metas, central_metas, right_metas


def deepbach_epoch_step(model, data, loss_func, optimizer, scheduler):
    device = torch.cuda.current_device()
    metrics = Metrics()
    start_time = time.time()

    subdivision = 4
    num_voices = 1
        
    for i, (d, label) in enumerate(data):
        
        for data in d:
            data = data.long().to(device = device, non_blocking = True)
        
        tensor_chorale, tensor_metadata = d
            

        
        for voice_model, voice_optimizer in zip(model.voice_models, optimizer.optimizers):
            if model.training:
                voice_optimizer.zero_grad()
            notes, metas, label = preprocess_input(tensor_chorale, tensor_metadata, voice_model.main_voice_index)
            notes = [note.to(device = device, non_blocking = True) for note in notes]
            metas = [meta.to(device = device, non_blocking = True) for meta in metas]
            label = label.to(device = device, non_blocking = True)
        
            weights = voice_model.forward(notes, metas)
            loss_function = torch.nn.CrossEntropyLoss()
            loss = loss_function(weights, label)
            
            
            y_pred = weights.reshape(-1, weights.size(-1)).max(-1)[-1]
            y_true = label.reshape(-1)

            if model.training:
                loss.backward()
                voice_optimizer.step()
                if scheduler != None:
                    scheduler.step()

            metrics.loss += loss.detach().item()
            metrics.calc_metrics(y_true.detach().to('cpu').numpy(), y_pred.detach().to('cpu').numpy())

        print_and_save_logs(metrics, model.training, i, len(data), start_time, save=False)
    
    return metrics



def __main__():
    model, optimizer, scheduler = init_anticipation_rnn_model()