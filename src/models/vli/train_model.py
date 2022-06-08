import numpy as np
import math
import sys
import time
import datetime
import os
import copy

from src.report.logs import print_and_save_logs
from src.report.metrics import Metrics

from .modified_xlnet import XLNetModel, XLNetConfig, AdamW

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

#import prepare_data
import pickle
from pathlib import Path

np.set_printoptions(threshold=sys.maxsize)

PROJECT_DIR = Path(__file__).resolve().parents[3]
MODELS_DIR = os.path.join(PROJECT_DIR, "models")


device = torch.cuda.current_device()

configuration = XLNetConfig().from_dict({
  "_name_or_path": "xlnet-predict-middle-notes",
  "architectures": [
    "XLNetLMHeadModel"
  ],
  "attn_type": "bi",
  "bi_data": False,
  "bos_token_id": 10000,
  "clamp_len": -1,
  # "d_head": 64,
  "d_inner": 3072,
  "d_model": 768,
  "dropout": 0.1,
  "end_n_top": 5,
  "eos_token_id": 2,
  "ff_activation": "gelu",
  "initializer_range": 0.02,
  "layer_norm_eps": 1e-12,
  "mem_len": None, # null
  "model_type": "xlnet",
  "n_head": 8,  # 12 originally
  "n_layer": 12,
  "pad_token_id": 10000,
  "reuse_len": None, # null,
  "same_length": False,
  "start_n_top": 5,
  "summary_activation": "tanh",
  "summary_last_dropout": 0.1,
  "summary_type": "last",
  "summary_use_proj": True,
  "untie_r": True,
  "use_mems_eval": True,
  "use_mems_train": True,
  # "vocab_size": 32000
})


class Embeddings(nn.Module):
    def __init__(self, n_token, d_model):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(n_token, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class XLNetForPredictingMiddleNotes(torch.nn.Module):
    def __init__(self, xlnet_config, e2w, w2e, is_train=None):
        super(XLNetForPredictingMiddleNotes, self).__init__()
        self.xlnet = XLNetModel(xlnet_config)
        self.xlnet_config = xlnet_config
        self.loss_func = nn.CrossEntropyLoss(reduction='none')

        # token types: [Tempo, Bar, Position, Pitch, Duration, Velocity]
        self.n_tokens = []
        for key in e2w:
            self.n_tokens.append(len(e2w[key]))
        # Use relative bar instead of absolute bar encoding
        self.n_tokens[1] = 4

        self.emb_sizes = [256, 256, 256, 256, 256, 256]
        self.e2w = e2w
        self.w2e = w2e

        # for deciding whether the current input_ids is a <PAD> token
        self.tempo_pad_word = self.e2w['Tempo']['Tempo <PAD>']

        self.eos_word = torch.tensor([self.e2w[etype]['%s <EOS>' % etype] for etype in self.e2w]).long().to(device)
        self.bos_word = torch.tensor([self.e2w[etype]['%s <BOS>' % etype] for etype in self.e2w]).long().to(device)
        self.mask_word = torch.tensor([0, 0, 0, 0, 0, 0]).long().to(device)

        # word_emb: embeddings to change token ids into embeddings
        self.word_emb = []
        for i, key in enumerate(self.e2w):
            self.word_emb.append(Embeddings(self.n_tokens[i], self.emb_sizes[i]))
        self.word_emb = nn.ModuleList(self.word_emb)

        # linear layer to merge embeddings from different token types to feed into xlnet
        self.in_linear = nn.Linear(np.sum(self.emb_sizes), xlnet_config.d_model)

        # proj: project embeddings to logits for prediction
        self.proj = []
        for i, etype in enumerate(self.e2w):
            self.proj.append(nn.Linear(xlnet_config.d_model, self.n_tokens[i]))
        self.proj = nn.ModuleList(self.proj)


    def forward(self, input_ids, attention_mask, perm_mask, target_mapping, bar_ids=None, input_ids_g=None):
        """
        Args:
            input_ids: of shape [bsz, seq_len, n_event_type]. Input for content stream.
        """
        # convert input_ids into embeddings and merge them through linear layer
        embs =[]
        for i, key in enumerate(self.e2w):
            embs.append(self.word_emb[i](input_ids[..., i]))
        embs = torch.cat([*embs], dim=-1)
        emb_linear = self.in_linear(embs)

        # (for query stream) convert input_ids into embeddings and merge them through linear layer
        embs_g =[]
        for i, key in enumerate(self.e2w):
            embs_g.append(self.word_emb[i](input_ids_g[..., i]))
        embs_g = torch.cat([*embs_g], dim=-1)
        emb_linear_g = self.in_linear(embs_g)

        # feed to xlnet
        y = self.xlnet(inputs_embeds=emb_linear,
                       attention_mask=attention_mask,
                       perm_mask=perm_mask,
                       target_mapping=target_mapping,
                       inputs_embeds_g=emb_linear_g,
                       bar_ids=bar_ids)
        y = y.last_hidden_state

        # convert embeddings back to logits for prediction
        ys = []
        for i, etype in enumerate(self.e2w):
            ys.append(self.proj[i](y))

        return ys


    def compute_loss(self, predict, target, loss_mask):
        loss = self.loss_func(predict, target)
        loss = loss * loss_mask
        loss = torch.sum(loss) / torch.sum(loss_mask)
        return loss

def construct_dict(fraction):
	event2word = {}
	word2event = {}
	tempo_quantize_step = 4
	ctxt_size = 16
	velocity_bins = 32

	for etype in ['Tempo', 'Bar', 'Position', 'Pitch', 'Duration', 'Velocity']:
		count = 0
		e2w = {}

		# Tempo 30 ~ 210
		if etype == 'Tempo':
			for i in range(28, 211, tempo_quantize_step):
				e2w['Tempo %d' % i] = count
				count += 1

		# Bar 0 ~ 15
		elif etype == 'Bar':
			for i in range(ctxt_size):
				e2w['Bar %d' % i] = count
				count += 1

		# Position: 0/16 ~ 15/16
		elif etype == 'Position':
			for i in range(0, fraction):
				e2w['Position %d/16' % i] = count
				count += 1

		# Pitch: 22 ~ 107
		elif etype == 'Pitch':
			for i in range(22, 108):
				e2w['Pitch %d' % i] = count
				count += 1

		# Duration: 0 ~ 63
		elif etype == 'Duration':
			for i in range(64):
				e2w['Duration %d' % i] = count
				count += 1

		# Velocity: 0 ~ 21
		elif etype == 'Velocity':
			for i in range(velocity_bins):
				e2w['Velocity %d' % i] = count
				count += 1

		else:
			raise Exception('etype error')


		e2w['%s <BOS>' % etype] = count
		count += 1
		e2w['%s <EOS>' % etype] = count
		count += 1
		e2w['%s <PAD>' % etype] = count
		count += 1

		event2word[etype] = e2w
		word2event[etype] = {e2w[key]: key for key in e2w}

	#print(event2word)
	return event2word, word2event

e2w, w2e = construct_dict(fraction=16)


def epoch_step(model, data, loss_func, optimizer, scheduler):
    # based on train method modified to accept DataLoader objects
    tempo_pad_word = e2w['Tempo']['Tempo <PAD>']
    eos_word = torch.tensor([e2w[etype]['%s <EOS>' % etype] for etype in e2w]).long().to(device)
    bos_word = torch.tensor([e2w[etype]['%s <BOS>' % etype] for etype in e2w]).long().to(device)
    target_max_percent = 0.25
    max_seq_len = 512
    start_time = time.time()
    metrics = Metrics()
    
        
    for it, (d, target) in enumerate(data):
        #breakpoint()
        input_ids = d.to(device)
        batch_size = d.shape[0]
        start_end_batch = []
        
        for i in range(batch_size):
            start = torch.nonzero(input_ids[i, :, 1] >= 6)[0]
            end = torch.nonzero(input_ids[i, :, 1] <= 9)[-1]
            start_end_batch.append([start.item(), end.item()])
        start_end_batch = torch.Tensor(start_end_batch)        

        attn_mask = (input_ids[:, :, 0] != tempo_pad_word).float()

        # decide the range to be predicted: `target_start` to `target_start + target_len`
        valid_seq_lengths = [torch.nonzero(seq)[-1][0] + 1 for seq in attn_mask] # seq length without <PAD> tokens
        target_starts = [np.random.randint(int(seq_len * (1 - target_max_percent))) for seq_len in valid_seq_lengths]
        target_lens = [np.random.randint(int(seq_len * target_max_percent / 2), int(seq_len * target_max_percent)) for seq_len in valid_seq_lengths]


        # generate perm_mask
        # 0: attend, 1: do not attend
        perm_mask = torch.ones(batch_size, max_seq_len, max_seq_len).to(device)
        perm_mask_ = torch.ones(batch_size, max_seq_len, max_seq_len).to(device)
        for b in range(batch_size):
            perm_mask[b, :, :target_starts[b]] = 0
            perm_mask[b, :, target_starts[b] + target_lens[b]:valid_seq_lengths[b]] = 0
            for i in range(target_starts[b], target_starts[b]+target_lens[b]):
                perm_mask[b, i, target_starts[b]:i] = 0

        #print(perm_mask)

        # target mapping: partial prediction
        target_mapping = torch.zeros(batch_size, max(target_lens), max_seq_len).to(device)
        for b in range(batch_size):
            for i, j in enumerate(range(target_starts[b], target_starts[b]+target_lens[b])):
                target_mapping[b, i, j] = 1

        #print("target_mapping", target_mapping)

        # change to use relative bar representation
        bar_ids = torch.clone(input_ids[:, :, 1]).detach()
        input_ids[:, 1:, 1] = input_ids[:, 1:, 1] - input_ids[:, :-1, 1]
        input_ids[:, :, 1][input_ids[:, :, 1] > 1] = 1  # avoid bug when there are empty bars

        #print(input_ids)

        input_ids_g = torch.zeros(batch_size, max(target_lens), len(e2w)).long().to(device)
        for b in range(batch_size):
            input_ids_g[b, :target_lens[b]] = input_ids[b, target_starts[b]:target_starts[b]+target_lens[b]]
            input_ids_g[b, :target_lens[b], [0, 3, 4, 5]] = bos_word[[0, 3, 4, 5]]  # mask out tokens except bar & pos
        
        #print(input_ids_g, input_ids_g.shape)
        y = model.forward(input_ids,
                                attn_mask,
                                perm_mask,
                                target_mapping,
                                bar_ids=bar_ids,
                                input_ids_g=input_ids_g)

        #print("yshape", len(y))

        for i, etype in enumerate(e2w):
                y[i] = y[i][:, ...].permute(0, 2, 1)

        #print(len(y))

        # calculate losses
        target = torch.zeros(batch_size, max(target_lens), len(e2w), dtype=torch.long).to(device)
        loss_mask = torch.zeros(batch_size, max(target_lens))
        for b in range(batch_size):
            target[b, :target_lens[b], [0, 3, 4, 5]] = input_ids[b, target_starts[b]:target_starts[b]+target_lens[b], [0, 3, 4, 5]]

            # next onset prediction
            target[b, :target_lens[b]-1, [1, 2]] = input_ids[b, target_starts[b]+1:target_starts[b]+target_lens[b], [1, 2]]
            target[b, target_lens[b]-1, 1] = 2  # <REL-BAR EOS>
            target[b, target_lens[b]-1, 2] = eos_word[2]

            loss_mask[b, :target_lens[b]] = 1

        losses = []
        for i, etype in enumerate(e2w):
            losses.append(model.compute_loss(y[i], target[..., i].to(device), loss_mask.to(device)))
        total_loss = sum(losses) / len(e2w)

        # udpate
        if model.training:
            model.zero_grad()
            total_loss.backward()
            clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
            if scheduler != None:
                scheduler.step()

        #print(y[4].argmax(dim=1).shape, target[..., 4].to(device).shape, loss_mask.to(device).shape)
        y_pred = [y_i.argmax(dim=1) for y_i in y]
        y_pred = torch.transpose(torch.stack(y_pred, dim=1), 1, 2)
        y_pred = y_pred.reshape(-1, y_pred.shape[-1])
        y_true = target.reshape(-1, target.shape[-1])
        loss_mask = torch.flatten(loss_mask).repeat(6,1).transpose(1,0).to(device)
        #print(y_pred.shape, y_true.shape, loss_mask.shape)
        y_pred = loss_mask * y_pred
        y_true = loss_mask * y_true 
        metrics.loss += total_loss.detach().item()
        metrics.calc_metrics(y_true.detach().to('cpu').numpy(), y_pred.detach().to('cpu').numpy())
        print_and_save_logs(metrics, model.training, it, len(data), start_time, save=False)
    
    return metrics


def generate(model, data):
    bos_word = torch.tensor([e2w[etype]['%s <BOS>' % etype] for etype in e2w]).long()
    

    # unpad
    tempo_pad_word = e2w['Tempo']['Tempo <PAD>']
    non_pad_indexes = torch.nonzero(torch.where(data[:,0] == tempo_pad_word, 0, 1))
    data = data[:non_pad_indexes[-1]]
    
    attn_mask = torch.ones_like(data[:,0])

    measures = data[:,1]
    perm_mask = torch.where(6 <= measures, 1, 0) + torch.where(measures <= 10, 1, 0) - 1
    inpaint_indexes = torch.nonzero(perm_mask)
    target_starts = inpaint_indexes[0]
    target_lens = [len(inpaint_indexes)]
    
    perm_mask = perm_mask.repeat([len(perm_mask),1])
    
    max_seq_len = len(data)
    target_mapping = torch.zeros(max(target_lens), max_seq_len)
    for i, j in enumerate(range(target_starts[0], target_starts[0]+target_lens[0])):
        target_mapping[i, j] = 1


    # change to use relative bar representation
    bar_ids = torch.clone(data[:, 1]).detach()
    data[1:, 1] = data[1:, 1] - data[:-1, 1]
    data[:, 1][data[:, 1] > 1] = 1  # avoid bug when there are empty bars

    input_ids_g = torch.zeros(max(target_lens), len(e2w)).long()
    input_ids_g[:target_lens[0]] = data[target_starts[0]:target_starts[0]+target_lens[0]]
    input_ids_g[:target_lens[0], [0, 3, 4, 5]] = bos_word[[0, 3, 4, 5]]  # mask out tokens except bar & pos
        
    y = model.forward(
        data.unsqueeze(0).to(device),
        attn_mask.unsqueeze(0).to(device),
        perm_mask.unsqueeze(0).to(device),
        target_mapping.unsqueeze(0).to(device),
        bar_ids=bar_ids.unsqueeze(0).to(device),
        input_ids_g=input_ids_g.unsqueeze(0).to(device)
    )
    for i, etype in enumerate(e2w):
        y[i] = y[i][:, ...].permute(0, 2, 1)
    y_pred = [y_i.argmax(dim=1) for y_i in y]
    y_pred = torch.transpose(torch.stack(y_pred, dim=1), 1, 2)

    #target = torch.zeros(1, max(target_lens), len(e2w), dtype=torch.long).to(device)
    #target[0, :target_lens[0], [0, 3, 4, 5]] = input_ids[0, target_starts[0]:target_starts[0]+target_lens[0], [0, 3, 4, 5]]

    # next onset prediction
    #target[0, :target_lens[0]-1, [1, 2]] = input_ids[0, target_starts[0]+1:target_starts[0]+target_lens[0], [1, 2]]
    #target[0, target_lens[0]-1, 1] = 2  # <REL-BAR EOS>
    #target[0, target_lens[0]-1, 2] = eos_word[2]
        

    return y_pred

def init_model(args):
    model = XLNetForPredictingMiddleNotes(configuration, e2w, w2e, is_train=True).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = None
        
    model_path = os.path.join(MODELS_DIR, f"{args.model}{args.mode}_{args.dataset}.pt")
    print(model_path)
    if os.path.isfile(model_path):
        print(f"Loading VLI from {model_path}")
        checkpoint = torch.load(model_path)
        
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        #scheduler.load_state_dict(checkpoint["scheduler"])

    if torch.cuda.is_available():
        print('Using: ', torch.cuda.get_device_name(torch.cuda.current_device()))
        model.cuda(torch.cuda.current_device())
    else:
        raise Exception("You are not using cuda GPU") 
    
    return model, optimizer, None


def nucleus(logit, p=0.9, t=1.2):
        logit = logit.cpu().detach().numpy()
        probs = temperature(logits=logit, temperature=t)
        cur_word = nucleus_(probs, p=p)
        return cur_word

def nucleus_(probs, p):
    probs /= sum(probs)
    sorted_probs = np.sort(probs)[::-1]
    sorted_index = np.argsort(probs)[::-1]
    cusum_sorted_probs = np.cumsum(sorted_probs)
    after_threshold = cusum_sorted_probs > p
    if sum(after_threshold) > 0:
        last_index = np.where(after_threshold)[0][0] + 1

        candi_index = sorted_index[:last_index]
    else:
        candi_index = sorted_index[:3] # just assign a value
    candi_probs = [probs[i] for i in candi_index]
    candi_probs /= sum(candi_probs)
    word = np.random.choice(candi_index, size=1, p=candi_probs)[0]
    return word


def temperature(logits, temperature):
    probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
    return probs

def predict(model, data):
    mask_word = torch.tensor([0, 0, 0, 0, 0, 0]).long().to(device)
    eos_word = torch.tensor([e2w[etype]['%s <EOS>' % etype] for etype in e2w]).long().to(device)
    bos_word = torch.tensor([e2w[etype]['%s <BOS>' % etype] for etype in e2w]).long().to(device)
    
    # unpad
    tempo_pad_word = e2w['Tempo']['Tempo <PAD>']
    non_pad_indexes = torch.nonzero(torch.where(data[:,0] == tempo_pad_word, 0, 1))
    data = data[:non_pad_indexes[-1]]
    data_ = data.clone().unsqueeze(0)
    datum = data.unsqueeze(0)
    
    seq_len = datum.shape[1]

    start_bar6 = np.nonzero(data_[0, :, 1] == 6)[0][0]
    end_bar9   = np.nonzero(data_[0, :, 1] == 10)[0][0]-1
    
    # target_len = np.random.randint(int((end_bar9 - start_bar6) * 0.5), end_bar9 - start_bar6 + 1)
    target_len = end_bar9 - start_bar6 +1
    # target_start = np.random.randint(start_bar6, end_bar9 - target_len + 1)
    target_start = start_bar6

    first_onset = data_[0, target_start, [1, 2]]
    first_onset_rel = np.copy(data_[0, target_start, [1, 2]])
    first_onset_rel[0] -= data_[0, target_start - 1, 1]
    #target_begin_token = [w2e[etype][datum[0, target_start, j]].split(' ')[1] for j, etype in enumerate(w2e)]
    #target_end_token = [w2e[etype][datum[0, target_start+target_len-1, j]].split(' ')[1] for j, etype in enumerate(w2e)]
    
    # Save prime
    #prime = np.concatenate([datum[0, :target_start], datum[0, target_start + target_len :]], axis=0)
    
    # Save absolute Bar IDs
    bar_ids_abs = np.copy(data_[:, :, 1])

    # abs -> rel Bar IDs
    datum[:, 1:, 1] = datum[:, 1:, 1] - datum[:, :-1, 1]
    datum[:, :, 1][datum[:, :, 1] > 1] = 1  # avoid bug when there are empty bars

    # A_C -> AC #parece que por aqui está el bug
    datum[:, target_start : seq_len - target_len] = datum[:, target_start + target_len :].clone()
    datum = datum[:, : seq_len - target_len]
    bar_ids_abs[:, target_start : seq_len - target_len] = bar_ids_abs[:, target_start + target_len :]
    bar_ids_abs = bar_ids_abs[:, : seq_len - target_len]

    
    input_ids = datum.to(device)
    bar_ids = torch.from_numpy(bar_ids_abs).to(device)

    next_bar_abs = torch.tensor(first_onset[0]).long().to(device)
    next_onset = torch.from_numpy(first_onset_rel).long().to(device)
    condition_len = input_ids.shape[1]
    attn_mask = None

    while True:
        input_ids = torch.cat([input_ids, mask_word[None, None]], dim=1)
        input_ids_g = torch.clone(bos_word)
        input_ids_g[[1, 2]] = next_onset
        input_ids_g = input_ids_g[None, None]
        bar_ids = torch.cat([bar_ids, next_bar_abs[None, None]], dim=-1)

        # generate perm_mask
        # 0: attend, 1: do not attend
        perm_mask = torch.ones(1, input_ids.shape[1], input_ids.shape[1]).to(device)
        perm_mask[0, :, :condition_len] = 0
        for i in range(condition_len, input_ids.shape[1]):
            perm_mask[0, i, condition_len:i] = 0

        # target mapping: partial prediction
        target_mapping = torch.zeros(1, 1, input_ids.shape[1]).to(device)
        target_mapping[0, 0, -1] = 1

        y = model.forward(input_ids,
                            attn_mask,
                            perm_mask,
                            target_mapping,
                            bar_ids=bar_ids,
                            input_ids_g=input_ids_g)

        # sampling
        y_logits = []
        for i, etype in enumerate(e2w):
            y_logits.append(y[i][0, -1, :])
        cur_word = []
        for i, etype in enumerate(e2w):
            cur_word.append(nucleus(y_logits[i], p=0.9, t=1.0))#t=0.8))
        cur_word = np.array(cur_word)

        input_ids[0, -1, [1, 2]] = next_onset
        input_ids[0, -1, [0, 3, 4, 5]] = torch.from_numpy(cur_word).to(device)[[0, 3, 4, 5]]
        next_onset = torch.from_numpy(cur_word).to(device)[[1, 2]]
        next_bar_abs = next_onset[0] + bar_ids[0, -1]

        # if 'EOS' in self.w2e['Bar'][cur_word[1]]:
        if cur_word[1] == 2:
            break
        if 'EOS' in w2e['Position'][cur_word[2]]:
            break
        if input_ids.shape[1] >= 1000:
            break

    
    input_ids = input_ids.cpu().detach().numpy()[0]
    bar_ids = bar_ids.cpu().detach().numpy()[0]
    input_ids[:, 1] = bar_ids
    out_start6 = np.nonzero(input_ids[:, 1] >= 6)[0]
    out_end9 = np.nonzero(input_ids[:, 1] < 10)[0]
    inpaint_index = list(set(out_start6) & set(out_end9))
    out_start6 = inpaint_index[0]
    out_end9 = inpaint_index[-1]+1
    return input_ids[out_start6:out_end9]