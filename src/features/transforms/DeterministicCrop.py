import torch
import copy
import numpy as np

class DeterministicCrop:
    def __init__(self, ctxt_size, fraction, model_name):
        self.ctxt_size = ctxt_size
        self.fraction = fraction
        self.model_name = model_name


    def __call__(self, data):
        # data: [n_instruments, time_steps] or [n_instruments, measures, n_notes, 6]
        n_measures = len(data[0])//self.fraction #ojo que el ultimo compas si no es de tamaño 24 muere
        high = n_measures+1-self.ctxt_size
        min_index = 0 #torch.randint(low=0, high=high, size=(1,))*self.fraction
        max_index = self.fraction*self.ctxt_size
        data = np.array(data)
        ctxt = data[:,min_index:max_index]
        if self.model_name == "SKETCHNET" or self.model_name == "SKETCHVAE" or self.model_name == "MEASUREVAE":
            ctxt = self.fix_empty_pitches(ctxt)
        
        return ctxt

    def fix_empty_pitches(self, data):
        n = len(data[0])//self.fraction
        for i in range(n):
            measure = data[0][i*self.fraction:(i+1)*self.fraction]
            if len(measure[measure<128]) == 0:
                data[0][i*self.fraction] = 60
        return data
