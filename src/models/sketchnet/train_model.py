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
from .sketchnet import SketchNet
from torch.nn import functional as F
from .utils import MinExponentialLR, processed_data_tensor, vae_processed_data_tensor, vae_loss_function, process_raw_x

from pathlib import Path
import os

PROJECT_DIR = Path(__file__).resolve().parents[3]
RAW_DIR = os.path.join(PROJECT_DIR, "data", "raw", "folk/")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")

hidden_dims = 1024
zp_dims = 128
zr_dims = 128
vae_beta = 0.1
input_dims = 130
pitch_dims = 129
rhythm_dims = 3
seq_len = 4 * 6
beat_num = 4
tick_num = 6

zp_dims = 128
zr_dims = 128
pf_dims = 512
gen_dims = 1024
combine_dims = 512
combine_head = 4
combine_num = 4
pf_num = 2
inpaint_len = 4
total_len = 16

n_past = 6
n_future = 10
n_inpaint = 4


def init_vae(args):
    model = SketchVAE(input_dims, pitch_dims, rhythm_dims, hidden_dims, zp_dims, zr_dims, seq_len, beat_num, tick_num, 4000)
    if torch.cuda.is_available():
        print('Using: ', torch.cuda.get_device_name(torch.cuda.current_device()))
        model.cuda(torch.cuda.current_device())
    else:
        raise Exception("You are not using cuda GPU") 
        
    optimizer = optim.Adam(model.parameters(), lr = args.vae_lr)
    scheduler = MinExponentialLR(optimizer, gamma = args.vae_decay, minimum = args.vae_min_decay)
    model_path = os.path.join(MODELS_DIR, f"{args.vae_model}_{args.model}_{args.dataset}.pt")
    if os.path.isfile(model_path):
        checkpoint = torch.load(model_path)
        
        print(f"Loading SketchVAE from {model_path}")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

    

    return model, optimizer, scheduler

def init_model(vae_model, args):
    model = SketchNet(zp_dims, zr_dims, pf_dims, gen_dims, combine_dims, pf_num, combine_num, combine_head, inpaint_len, total_len, vae_model, True)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=args.lr)
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

        label = d["inpaint_gd_whole"] #label.to(device = device, non_blocking = True)
        
        if model.training:
            optimizer.zero_grad()
        
        recon_x, iteration, use_teacher, stage = model(d["past_x"], d["future_x"], d["middle_x"])
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

    
    recon_x, iteration, use_teacher, stage = model(d["past_x"], d["future_x"], d["middle_x"])
    y_pred = recon_x.view(-1, recon_x.size(-1)).max(-1)[-1]
    return y_pred  
