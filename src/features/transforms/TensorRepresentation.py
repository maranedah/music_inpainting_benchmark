import torch
class TensorRepresentation:
    def __init__(self, filter_instruments):
        self.filter_instruments = filter_instruments


    def __call__(self, data):
        if self.filter_instruments == None:
            data = [x["notes"] for x in data]
        else:
            data = [x["notes"] for x in data if x["name"] in self.filter_instruments]
        return data    
        