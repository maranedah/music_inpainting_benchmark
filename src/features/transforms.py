import torch
from torch.nn.functional import one_hot

__all__ = ["Compose", "FixEmptyMeasures", "Factorize"]


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, midi):
        for t in self.transforms:
            midi = t(midi)
        return midi


class FixEmptyMeasures:
    def __init__(self, resolution):
        self.resolution = resolution

    def __call__(self, data):
        n = len(data) // self.resolution
        for i in range(n):
            measure = data[i * self.resolution : (i + 1) * self.resolution]
            if len(measure[measure < 128]) == 0:
                data[i * self.resolution] = 60
        return data


class Factorize:
    def __init__(self, ctxt_split=(6, 4, 6), resolution=24):
        self.ctxt_split = ctxt_split
        self.resolution = resolution

    def __call__(self, data):
        data = torch.Tensor(data).squeeze(0)
        if len(self.ctxt_split) == 1:
            re = self.factorize(data, self.resolution)
            re = [x.squeeze(0) for x in re]
        elif len(self.ctxt_split) == 3:
            ctxt_resolution = tuple([x * self.resolution for x in self.ctxt_split])
            past, middle, future = data.split(ctxt_resolution)
            past_x = self.factorize(past, self.resolution)
            middle_x = self.factorize(middle, self.resolution)
            future_x = self.factorize(future, self.resolution)
            re = {
                "inpaint_gd_whole": middle_x[4].contiguous().view(-1),  # middle_gd
                "past_x": past_x,
                "middle_x": middle_x,
                "future_x": future_x,
            }
        return re

    def factorize(self, data, resolution):
        ones = torch.ones_like(data) * 127
        rx = torch.where(data < 128, ones, data) - 126
        nrx = torch.stack(
            one_hot((rx - 1).to(torch.int64), num_classes=3).split(resolution)
        )
        rx = torch.stack(rx.split(resolution))
        gd = torch.stack(data.split(resolution))
        len_x = torch.Tensor([len(m[m < 128]) for m in gd])
        px = [m[m < 128] for m in gd]
        px = torch.stack(
            [torch.cat((x, torch.ones(resolution - len(x)) * 128)) for x in px]
        )
        return [
            px.to(torch.long),
            rx.to(torch.float),
            len_x.to(torch.long),
            nrx.to(torch.float),
            gd.to(torch.long),
        ]
