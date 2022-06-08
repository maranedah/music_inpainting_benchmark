import torch
class RandomInstrument:
    def __init__(self, is_train):
        self.is_train = is_train


    def __call__(self, data):
        n_instruments = len(data) # data = [n_instruments, time_steps]
        if self.is_train:
            inst_index = torch.randint(low=0, high=n_instruments, size=(1,))
        else:
            inst_index = 0
        inst_data = [data[inst_index]]
        return inst_data