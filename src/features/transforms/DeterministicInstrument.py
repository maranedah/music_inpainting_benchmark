import torch
class DeterministicInstrument:


    def __call__(self, data):
        n_instruments = len(data) # data = [n_instruments, time_steps]
        inst_index = 0
        inst_data = [data[inst_index]]
        return inst_data