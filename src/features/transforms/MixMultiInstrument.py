import torch
import numpy as np
class MixMultiInstrument:
    def __call__(self, data):
            
        max_measure = max([len(inst) for inst in data])
        measures = [[] for i in range(max_measure)]
        for inst in data:
            for i in range(len(inst)):
                for note in inst[i]:
                    measures[i].append(note)
                measures[i].sort(key=lambda x: x[2])

        measures = [measures]
        return measures