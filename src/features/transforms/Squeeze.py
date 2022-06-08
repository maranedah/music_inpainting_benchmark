import torch
class Squeeze:
    def __init__(self, dim):
        self.dim = dim


    def __call__(self, data):
        return data[0]    
        