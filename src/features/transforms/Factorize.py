import torch
from torch.nn.functional import one_hot
class Factorize:
    def __init__(self, ctxt_split=(6, 4, 6), split_size=24):
        self.ctxt_split = ctxt_split
        self.split_size = split_size

    def __call__(self, data):
        data = torch.Tensor(data).squeeze(0)
        if len(self.ctxt_split) == 1:
            re = self.factorize(data, self.split_size)
            re = [x.squeeze(0) for x in re]
        elif len(self.ctxt_split) == 3:
            ctxt_split_size = tuple([x*self.split_size for x in self.ctxt_split])
            past, middle, future = data.split(ctxt_split_size)
            past_x = self.factorize(past, self.split_size)
            middle_x = self.factorize(middle, self.split_size)
            future_x = self.factorize(future, self.split_size)
            re = {
                "inpaint_gd_whole": middle_x[4].contiguous().view(-1), #middle_gd
                "past_x": past_x,
                "middle_x": middle_x,
                "future_x": future_x,
            }
        return re


    def factorize(self, data, split_size):
        ones = torch.ones_like(data) * 127
        rx = torch.where(data < 128, ones, data) - 126
        nrx = torch.stack(one_hot((rx-1).to(torch.int64), num_classes=3).split(split_size))
        rx = torch.stack(rx.split(split_size))
        gd = torch.stack(data.split(split_size))
        len_x = torch.Tensor([len(m[m<128]) for m in gd])
        px = [m[m<128] for m in gd]
        px = torch.stack([torch.cat((x, torch.ones(split_size-len(x))*128)) for x in px])
        return [px.to(torch.long), rx.to(torch.float), len_x.to(torch.long), nrx.to(torch.float), gd.to(torch.long)]

    

