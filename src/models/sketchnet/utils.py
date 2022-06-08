import numpy as np
import torch
from torch.utils.data import TensorDataset

from torch.nn import functional as F

from torch.distributions import kl_divergence, Normal
from torch.optim.lr_scheduler import ExponentialLR

class MinExponentialLR(ExponentialLR):
    def __init__(self, optimizer, gamma, minimum, last_epoch=-1):
        self.min = minimum
        super(MinExponentialLR, self).__init__(optimizer, gamma, last_epoch=-1)

    def get_lr(self):
        return [
            max(base_lr * self.gamma**self.last_epoch, self.min)
            for base_lr in self.base_lrs
        ]


def vae_processed_data_tensor(data, seq_len, rhythm_dims):
    print("processed data:")
    gd = np.array([d[0] for d in data])
    px = np.array([d[1] for d in data])
    rx = np.array([d[2] for d in data])
    len_x = np.array([d[3] for d in data])
    nrx = []
    for i,r in enumerate(rx):
        temp = np.zeros((seq_len, rhythm_dims))
        lins = np.arange(0, len(r))
        temp[lins, r - 1] = 1
        nrx.append(temp)
    nrx = np.array(nrx)
    gd = torch.from_numpy(gd).long()
    px = torch.from_numpy(px).long()
    rx = torch.from_numpy(rx).float()
    len_x = torch.from_numpy(len_x).long()
    nrx = torch.from_numpy(nrx).float()
    print("processed finish!")
    return TensorDataset(px, rx, len_x, nrx, gd)

def processed_data_tensor(data, vae_seq_len, vae_rhythm_dims):
    print("processed data:")
    gd = [] 
    px = []
    rx = []
    len_x = []
    nrx = []
    total = 0
    for i, d in enumerate(data):
        gd.append([list(dd[0]) for dd in d])
        px.append([list(dd[1]) for dd in d])
        rx.append([list(dd[2]) for dd in d])
        len_x.append([dd[3] for dd in d])
        if len(gd[-1][-1]) != vae_seq_len:
            gd[-1][-1].extend([128] * (vae_seq_len - len(gd[-1][-1])))
            px[-1][-1].extend([128] * (vae_seq_len - len(px[-1][-1])))
            rx[-1][-1].extend([2] * (vae_seq_len - len(rx[-1][-1])))
    for i,d in enumerate(len_x):
        for j,dd in enumerate(d):
            if len_x[i][j] == 0:
                gd[i][j][0] = 60
                px[i][j][0] = 60
                rx[i][j][0] = 1
                len_x[i][j] = 1
                total += 1
    gd = np.array(gd)
    px = np.array(px)
    rx = np.array(rx)
    len_x = np.array(len_x)
    for d in rx:
        nnrx = []
        for dd in d:
            temp = np.zeros((vae_seq_len, vae_rhythm_dims))
            lins = np.arange(0, len(dd))
            temp[lins, dd - 1] = 1
            nnrx.append(temp)
        nrx.append(nnrx)
    nrx = np.array(nrx)
    gd = torch.from_numpy(gd).long()
    px = torch.from_numpy(px).long()
    rx = torch.from_numpy(rx).float()
    len_x = torch.from_numpy(len_x).long()
    nrx = torch.from_numpy(nrx).float()
    print("processed finish! zeros:", total)
    print(gd.size(),px.size(),rx.size(),len_x.size(),nrx.size())
    return TensorDataset(px, rx, len_x, nrx, gd)

def process_raw_x(raw_x, n_past, n_inpaint, n_future):
    print(raw_x)
    raw_px, raw_rx, raw_len_x, raw_nrx, raw_gd = raw_x
    past_px = raw_px[:,:n_past,:]
    inpaint_px = raw_px[:,n_past:n_past + n_inpaint,:]
    future_px = raw_px[:,n_future:,:]
    past_rx = raw_rx[:,:n_past,:]
    inpaint_rx = raw_rx[:,n_past:n_past + n_inpaint,:]
    future_rx = raw_rx[:,n_future:,:]
    past_len_x = raw_len_x[:,:n_past]
    inpaint_len_x = raw_len_x[:,n_past:n_past + n_inpaint]
    future_len_x = raw_len_x[:,n_future:]
    past_nrx = raw_nrx[:,:n_past,:]
    inpaint_nrx = raw_nrx[:,n_past:n_past + n_inpaint,:]
    future_nrx = raw_nrx[:,n_future:,:]
    past_gd = raw_gd[:,:n_past,:]
    inpaint_gd = raw_gd[:,n_past:n_past + n_inpaint,:]
    future_gd = raw_gd[:,n_future:,:]
    re = [
        past_px, past_rx, past_len_x, past_nrx, past_gd,
        inpaint_px, inpaint_rx, inpaint_len_x, inpaint_nrx, inpaint_gd,
        future_px, future_rx, future_len_x, future_nrx, future_gd,
    ]
    return re

# loss function
def std_normal(shape):
    N = Normal(torch.zeros(shape), torch.ones(shape))
    if torch.cuda.is_available():
        N.loc = N.loc.cuda()
        N.scale = N.scale.cuda()
    return N

def vae_loss_function(recon, target, p_dis, r_dis, beta):
    CE = F.cross_entropy(recon.view(-1, recon.size(-1)), target, reduction = "mean")
    normal1 = std_normal(p_dis.mean.size())
    normal2=  std_normal(r_dis.mean.size())
    KLD1 = kl_divergence(p_dis, normal1).mean()
    KLD2 = kl_divergence(r_dis, normal2).mean()
    max_indices = recon.view(-1, recon.size(-1)).max(-1)[-1]
    correct = max_indices == target
    acc = torch.sum(correct.float()) / target.size(0)
    return acc, CE + beta * (KLD1 + KLD2)
