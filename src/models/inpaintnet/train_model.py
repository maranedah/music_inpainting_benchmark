
import torch
import os
import numpy as np
import pandas as pd
import sys
import time

from src.report.logs import print_and_save_logs
from src.report.metrics import Metrics

import argparse

from torch import optim
from torch.nn import functional as F
from torch import nn, distributions
#from torch.optim.lr_scheduler import ExponentialLR, CyclicLR

from pathlib import Path
import os

from .inpaint_rnn import LatentRNN
from .measure_vae import MeasureVAE


PROJECT_DIR = Path(__file__).resolve().parents[3]
MODELS_DIR = os.path.join(PROJECT_DIR, "models")

def init_vae(args):
    model = MeasureVAE()
    if torch.cuda.is_available():
        print('Using: ', torch.cuda.get_device_name(torch.cuda.current_device()))
        model.cuda(torch.cuda.current_device())
    else:
        raise Exception("You are not using cuda GPU")
        
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = None

    model_path = os.path.join(MODELS_DIR, f"{args.vae_model}_{args.model}_{args.dataset}.pt")
    if os.path.isfile(model_path):
        checkpoint = torch.load(model_path)
        
        print(f"Loading SketchVAE from {model_path}")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler:
            scheduler.load_state_dict(checkpoint["scheduler"])

    return model, optimizer, scheduler

def init_model(vae_model, args):
    model = LatentRNN(
        vae_model,
        num_rnn_layers=args.num_latent_rnn_layers,
        rnn_hidden_size=args.latent_rnn_hidden_size,
        dropout=args.latent_rnn_dropout_prob,
        rnn_class=torch.nn.GRU,
        auto_reg=args.auto_reg,
        teacher_forcing=args.teacher_forcing
        )
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

def mean_crossentropy_loss_alt(weights, targets):
        """
        Evaluates the cross entropy loss
        :param weights: torch Variable,
                (batch_size, num_measures, seq_len, num_notes)
        :param targets: torch Variable,
                (batch_size, num_measures, seq_len)
        :return: float, loss
        """
        criteria = torch.nn.CrossEntropyLoss(reduction='elementwise_mean')
        _, _, _, num_notes = weights.size()
        weights = weights.reshape(-1, num_notes)
        targets = targets.reshape(-1)
        loss = criteria(weights, targets)
        return loss

def epoch_step(model, data, loss_func, optimizer, scheduler):
    device = torch.cuda.current_device()
    metrics = Metrics()
    start_time = time.time()
        
    for i, (d, label) in enumerate(data):
        
            
        d = d.long()
        d = d.to(device = device,non_blocking = True)

        if model.training:
            optimizer.zero_grad()
        
        # extract data
        tensor_past, tensor_future, tensor_target = d[:,0:24*6].reshape(-1, 6, 24), d[:,24*10:].reshape(-1, 6, 24), d[:,24*6:24*10].reshape(-1, 4, 24),
        
        num_measures_past = tensor_past.size(1)
        num_measures_future = tensor_future.size(1)
        # perform forward pass of model
        weights, pred, _ = model(
            past_context=tensor_past,
            future_context=tensor_future,
            target=tensor_target,
            measures_to_generate=4,
            train=True
        )
        # compute loss
        loss = mean_crossentropy_loss_alt(
            weights=weights,
            targets=tensor_target
        )
        
        y_pred = weights.reshape(-1, weights.size(-1)).max(-1)[-1]
        y_true = tensor_target.reshape(-1)
        
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


def mean_crossentropy_loss(weights, targets):
        """
        Evaluates the cross entropy loss
        :param weights: torch Variable,
                (batch_size, seq_len, num_notes)
        :param targets: torch Variable,
                (batch_size, seq_len)
        :return: float, loss
        """
        criteria = nn.CrossEntropyLoss(reduction='elementwise_mean')
        batch_size, seq_len, num_notes = weights.size()
        assert (batch_size == targets.size(0))
        assert (seq_len == targets.size(1))
        weights = weights.contiguous().view(-1, num_notes)
        targets = targets.contiguous().view(-1)
        loss = criteria(weights, targets)
        return loss

def compute_kld_loss(z_dist, prior_dist, beta=0.001):
        """
        :param z_dist: torch.nn.distributions object
        :param prior_dist: torch.nn.distributions
        :param beta:
        :return: kl divergence loss
        """
        kld = distributions.kl.kl_divergence(z_dist, prior_dist)
        kld = beta * kld.sum(1).mean()
        return kld


def vae_epoch_step(model, data, loss_func, optimizer, scheduler):

    device = torch.cuda.current_device()
    model = model.to(torch.cuda.current_device())
    metrics = Metrics()
    start_time = time.time()
        
    for i, (d, target) in enumerate(data):
        
        d = d.long()
        d = d.to(device = device,non_blocking = True)

        if model.training:
            optimizer.zero_grad()
        
        weights, samples, z_dist, prior_dist, z_tilde, z_prior = model(
            measure_score_tensor=d,
            train=True
        )
        recons_loss = mean_crossentropy_loss(weights=weights, targets=d)
        dist_loss = compute_kld_loss(z_dist, prior_dist)
        loss = recons_loss + dist_loss
        
        y_pred = weights.argmax(dim=-1).view(-1)
        y_true = d.view(-1)
        
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

def generate(model, d):
    device = torch.cuda.current_device()
    d = torch.from_numpy(d)
    d = d.long().unsqueeze(0)
    d = d.to(device = device,non_blocking = True)
    
    # extract data
    tensor_past, tensor_future, tensor_target = d[:,0:24*6].reshape(-1, 6, 24), d[:,24*10:].reshape(-1, 6, 24), d[:,24*6:24*10].reshape(-1, 4, 24),
    
    # perform forward pass of model
    weights, pred, _ = model(
        past_context=tensor_past,
        future_context=tensor_future,
        target=tensor_target,
        measures_to_generate=4,
        train=True
    )
    
    y_pred = weights.reshape(-1, weights.size(-1)).max(-1)[-1]
    #y_true = tensor_target.reshape(-1)
    
    return y_pred  
