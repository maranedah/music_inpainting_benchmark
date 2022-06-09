import torch
import copy
import numpy as np
class RandomCrop:
    def __init__(self, ctxt_size, fraction, model_name, is_train):
        self.ctxt_size = ctxt_size
        self.fraction = fraction
        self.model_name = model_name
        self.is_train = is_train


    def __call__(self, data):
        # data: [n_instruments, time_steps] or [n_instruments, measures, n_notes, 6]
        n_measures = len(data[0])//self.fraction #ojo que el ultimo compas si no es de tamaño 24 muere
        high = n_measures+1-self.ctxt_size
        if self.is_train:
            min_index = torch.randint(low=0, high=high, size=(1,))*self.fraction
        else:
            min_index = 0
        max_index = min_index + self.fraction*self.ctxt_size
        data = np.array(data)
        ctxt = data[:,min_index:max_index]
        
        return ctxt

