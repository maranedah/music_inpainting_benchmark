import torch
import copy
import numpy as np
class FixEmptyPitches:
    def __init__(self, fraction):
        self.fraction = fraction
        

    def __call__(self, data):
        n = len(data[0])//self.fraction
        for i in range(n):
            measure = data[0][i*self.fraction:(i+1)*self.fraction]
            if len(measure[measure<128]) == 0:
                data[0][i*self.fraction] = 60
        return data
