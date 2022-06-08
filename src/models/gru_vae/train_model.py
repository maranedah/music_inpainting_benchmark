#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''
In this file 
We train our proposed SketchVAE
'''
import torch
import os
import numpy as np
import pandas as pd
import sys
import time

from dotenv import find_dotenv, load_dotenv

import logging
import argparse

from src.report.logs import print_and_save_logs
from src.report.metrics import Metrics

from torch import optim
from .sketchvae import SketchVAE
from .gru_vae import GRU_VAE
from torch.optim.lr_scheduler import ExponentialLR, CyclicLR

from pathlib import Path
import os

PROJECT_DIR = Path(__file__).resolve().parents[3]
RAW_DIR = os.path.join(PROJECT_DIR, "data", "raw", "folk/")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")

class MinExponentialLR(ExponentialLR):
    def __init__(self, optimizer, gamma, minimum, last_epoch=-1):
        self.min = minimum
        super(MinExponentialLR, self).__init__(optimizer, gamma, last_epoch=-1)

    def get_lr(self):
        return [
            max(base_lr * self.gamma**self.last_epoch, self.min)
            for base_lr in self.base_lrs
        ]

vae_beta = 0.9

def init_vae(args):
    model = SketchVAE(
        input_dims = 130,
        p_input_dims = 129,
        r_input_dims = 3,
        hidden_dims = 1024,
        zp_dims = 128,
        zr_dims = 128,
        seq_len = 24,
        beat_num = 4,
        tick_num = 6,
        decay = 4000
    )
    if torch.cuda.is_available():
        print('Using: ', torch.cuda.get_device_name(torch.cuda.current_device()))
        model.cuda(torch.cuda.current_device())
    else:
        raise Exception("You are not using cuda GPU") 
        
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    scheduler = CyclicLR(optimizer, base_lr=0.0001, max_lr=0.0006, step_size_up=600, step_size_down=2000, mode='exp_range', gamma=0.9997, cycle_momentum=False)
    model_path = os.path.join(MODELS_DIR, f"{args.vae_model}_{args.model}_{args.dataset}.pt")
    if os.path.isfile(model_path):
        checkpoint = torch.load(model_path)
        
        print(f"Loading SketchVAE from {model_path}")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])


    return model, optimizer, scheduler

def init_model(vae_model, args):
    model = GRU_VAE(
        zp_dims = 128,
        zr_dims = 128,
        pf_num = 2,
        inpaint_len = 4,
        gen_dims = 256,
        vae_model = vae_model
    )
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    scheduler = CyclicLR(optimizer, base_lr=0.0001, max_lr=0.0006, step_size_up=600, step_size_down=2000, mode='exp_range', gamma=0.9997, cycle_momentum=False)
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
        scheduler.load_state_dict(checkpoint["scheduler"])

    
    return model, optimizer, None 


def epoch_step(model, data, loss_func, optimizer, scheduler):
    device = torch.cuda.current_device()
    metrics = Metrics()
    start_time = time.time()
        
    for i, (d, label) in enumerate(data):

        for key, value in d.items():
            if isinstance(d[key], list):
                for j in range(len(d[key])):
                    d[key][j] = d[key][j].to(device = device, non_blocking = True)
            else:
                d[key] = d[key].to(device = device, non_blocking = True)

        label = d["inpaint_gd_whole"]

        if model.training:
            optimizer.zero_grad()
        
        recon_x = model(d["past_x"], d["future_x"], d["middle_x"])
        loss = loss_func(recon_x.view(-1, recon_x.size(-1)), label.view(-1), reduction = "mean")
        
        y_pred = recon_x.view(-1, recon_x.size(-1)).max(-1)[-1]
        y_true = label.view(-1)
        
        if model.training:
            loss.backward()
            optimizer.step()
            if scheduler != None:
                scheduler.step()

        metrics.loss += loss.detach().item()
        y_true = y_true.reshape(64, -1)
        y_pred = y_pred.reshape(64, -1)
        metrics.calc_metrics(y_true[0].detach().to('cpu').numpy(), y_pred[0].detach().to('cpu').numpy())

        print_and_save_logs(metrics, model.training, i, len(data), start_time, save=False)
    
    return metrics

def vae_epoch_step(model, data, loss_func, optimizer, scheduler):
    device = torch.cuda.current_device()
    metrics = Metrics()
    start_time = time.time()
        
    for i, (d, target) in enumerate(data):
        px, rx, len_x, nrx, gd = d
        
        px = px.to(device = device,non_blocking = True)
        len_x = len_x.to(device = device,non_blocking = True)
        nrx = nrx.to(device = device,non_blocking = True)
        gd = gd.to(device = device,non_blocking = True)
        target = gd

        if model.training:
            optimizer.zero_grad()
        
        recon, p_dis, r_dis, iteration = model(px, nrx, len_x, gd)
        acc, loss = loss_func(recon, target.view(-1), p_dis, r_dis, vae_beta)
        
        y_pred = recon.argmax(dim=-1).view(-1)
        y_true = target.view(-1)
        
        if model.training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            if scheduler != None:
                scheduler.step()

        metrics.loss += loss.detach().item()

        y_true = y_true.reshape(64, -1)
        y_pred = y_pred.reshape(64, -1)
        metrics.calc_metrics(y_true[0].detach().to('cpu').numpy(), y_pred[0].detach().to('cpu').numpy())


        print_and_save_logs(metrics, model.training, i, len(data), start_time, save=False)
    
    return metrics

def generate(model, d):
    device = torch.cuda.current_device()
    for key, value in d.items():
        if isinstance(d[key], list):
            for j in range(len(d[key])):
                d[key][j] = d[key][j].unsqueeze(0).to(device = device, non_blocking = True)
        else:
            d[key] = d[key].unsqueeze(0).to(device = device, non_blocking = True)

    
    recon_x = model(d["past_x"], d["future_x"], d["middle_x"])
    y_pred = recon_x.view(-1, recon_x.size(-1)).max(-1)[-1]
    return y_pred  
