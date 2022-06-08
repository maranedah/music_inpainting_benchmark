import torch
import pandas as pd
import os
from .encoded_midi import EncodedMidi
from .transforms.Compose import Compose
from .transforms.RandomCrop import RandomCrop
from .transforms.DeterministicCrop import DeterministicCrop
from .transforms.RandomTranspose import RandomTranspose
from .transforms.Factorize import Factorize
from .transforms.MixMultiInstrument import MixMultiInstrument
from .transforms.RandomInstrument import RandomInstrument
from .transforms.PadWords import PadWords
from .transforms.TensorRepresentation import TensorRepresentation
from .transforms.AssignBarNumbers import AssignBarNumbers
from .transforms.DeterministicInstrument import DeterministicInstrument
from .transforms.Identity import Identity


class SketchVaeDataset:
    def __init__(self, path: str, metadata: pd.DataFrame, dataset_name: str, split: str, mode: str, repeat=1):
        metadata = pd.concat([metadata]*repeat, axis=0)
        self.metadata = metadata.reset_index(drop=True)
        self.path = path
        self.model_name = "SKETCHVAE"
        self.dataset_name = dataset_name
        self.split = split
        if self.split=="train":
            self.transform = Compose(
                    [
                        TensorRepresentation(filter_instruments=None),
                        RandomInstrument(keep_instruments="all"),
                        RandomCrop(ctxt_size=1, fraction=24, model_name=self.model_name),
                        RandomTranspose(bounds=(0,127), mode="noteseq"), #TODO: bounds por min/max en el dataset (habria que construirlo mientras se hacen los frames) 
                        Factorize((1,))
                    ]
                )
        else:
            self.transform = Compose(
                    [
                        TensorRepresentation(filter_instruments=None),
                        DeterministicInstrument(),
                        DeterministicCrop(ctxt_size=1, fraction=24, model_name=self.model_name),
                        Factorize((1,))
                    ]
                )
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self,idx):
        data = EncodedMidi.from_path(
            file = os.path.join(self.path, self.metadata["filename"][idx]),
            process = "noteseq",
            dataset_name = self.dataset_name,
            fraction = 24,
            min_length = 1).get_tensor()
        data = self.transform(data)
        label = data[4] 
        return (data, label)

class SketchNetDataset:
    def __init__(self, path: str, metadata: pd.DataFrame, dataset_name: str, split: str, mode: str, repeat=1):
        metadata = pd.concat([metadata]*repeat, axis=0)
        self.metadata = metadata.reset_index(drop=True)
        self.path = path
        self.model_name = "SKETCHNET"
        self.dataset_name = dataset_name
        self.split = split
        self.cache = {}
        self.transform = Compose(
                [
                    TensorRepresentation(filter_instruments=None),
                    RandomInstrument(keep_instruments="all") if self.split=="train" else DeterministicInstrument(),
                    RandomCrop(ctxt_size=16, fraction=24, model_name=self.model_name) if self.split=="train" else DeterministicCrop(ctxt_size=16, fraction=24, model_name=self.model_name), 
                    RandomTranspose(bounds=(0,127), mode="noteseq"), #TODO: bounds por min/max en el dataset (habria que construirlo mientras se hacen los frames) 
                    Factorize(ctxt_split=(6, 4, 6), split_size=24)
                ]
            )
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self,idx):
        if not self.metadata["filename"][idx] in self.cache:
            data = EncodedMidi.from_path(
                file = os.path.join(self.path, self.metadata["filename"][idx]),
                process = "noteseq",
                dataset_name = self.dataset_name,
                fraction = 24,
                min_length = 16).get_tensor()
            self.cache[self.metadata["filename"][idx]] = data
        else:
            data = self.cache[self.metadata["filename"][idx]]
        data = self.transform(data)
        label = data["inpaint_gd_whole"]
        return (data, label)

class VLIDataset:
    def __init__(self, path: str, metadata: pd.DataFrame, dataset_name: str, split: str, mode: str, repeat=1):
        metadata = pd.concat([metadata]*repeat, axis=0)
        self.metadata = metadata.reset_index(drop=True)
        self.path = path
        self.model_name = "VLI"
        self.dataset_name = dataset_name
        self.split = split
        self.mode = mode
        self.cache = {}
        print("mode", self.mode)
        print("split", split)
        if self.split == "train":
            self.transform = Compose(
                    [
                        TensorRepresentation(filter_instruments=None),
                        MixMultiInstrument() if self.mode == "polyphony" else Identity(),
                        RandomInstrument(keep_instruments="all") if self.mode == "monophony" else Identity(),
                        RandomCrop(ctxt_size=16, fraction=1, model_name=self.model_name, is_train= self.split=="train"),  #fraction=1 es pq no hay split_size, cada measures un elemento del arreglo de data *buscar mejor nombre*
                        AssignBarNumbers(ctxt_size=16),
                        RandomTranspose(bounds=(0,127), representation="remi", is_train= self.split=="train"), #TODO: bounds por min/max en el dataset (habria que construirlo mientras se hacen los frames) 
                        PadWords(fraction=16),
                        
                    ]
                )
        else:
            self.transform = Compose(
                    [
                        TensorRepresentation(filter_instruments=None),
                        MixMultiInstrument() if self.mode == "polyphony" else Identity(),
                        DeterministicInstrument() if self.mode == "monophony" else Identity(),
                        DeterministicCrop(ctxt_size=16, fraction=1, model_name=self.model_name),  #fraction=1 es pq no hay split_size, cada measures un elemento del arreglo de data *buscar mejor nombre*
                        AssignBarNumbers(ctxt_size=16),
                        PadWords(fraction=16),
                    ]
                )            

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self,idx):
        if not self.metadata["filename"][idx] in self.cache:
            data = EncodedMidi.from_path(
                file = os.path.join(self.path, self.metadata["filename"][idx]),
                process = "remi",
                dataset_name = self.dataset_name,
                fraction = 16,
                min_length = 16).get_tensor()
            self.cache[self.metadata["filename"][idx]] = data
        else:
            data = self.cache[self.metadata["filename"][idx]]
        
        data = torch.tensor(self.transform(data))
        label = "bla"
        return (data, label)



class MeasureVaeDataset:
    def __init__(self, path: str, metadata: pd.DataFrame, dataset_name: str, split: str, mode: str, repeat=1):
        metadata = pd.concat([metadata]*repeat, axis=0)
        self.metadata = metadata.reset_index(drop=True)
        self.path = path
        self.model_name = "MEASUREVAE"
        self.dataset_name = dataset_name
        self.split = split
        self.transform = Compose(
                [
                    TensorRepresentation(filter_instruments=None),
                    RandomInstrument(keep_instruments="all") if self.split=="train" else DeterministicInstrument(),
                    RandomCrop(ctxt_size=1, fraction=24, model_name=self.model_name) if self.split=="train" else DeterministicCrop(ctxt_size=1, fraction=24, model_name=self.model_name),
                    
                ]
            )
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self,idx):
        data = EncodedMidi.from_path(
            file = os.path.join(self.path, self.metadata["filename"][idx]),
            process = "noteseq",
            dataset_name = self.dataset_name,
            fraction = 24,
            min_length = 1).get_tensor()
        data = self.transform(data)[0] #required unsqueeze
        label = "bla" 
        return (data, label)


class InpaintNetDataset:
    def __init__(self, path: str, metadata: pd.DataFrame, dataset_name: str, split: str, mode: str, repeat=1):
        metadata = pd.concat([metadata]*repeat, axis=0)
        self.metadata = metadata.reset_index(drop=True)
        self.path = path
        self.model_name = "INPAINTNET"
        self.dataset_name = dataset_name
        self.split = split
        self.transform = Compose(
                [
                    TensorRepresentation(filter_instruments=None),
                    RandomInstrument(keep_instruments="all") if self.split=="train" else DeterministicInstrument(),
                    RandomCrop(ctxt_size=16, fraction=24, model_name=self.model_name) if self.split=="train" else DeterministicCrop(ctxt_size=16, fraction=24, model_name=self.model_name), 
                ]
            )
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self,idx):
        data = EncodedMidi.from_path(
            file = os.path.join(self.path, self.metadata["filename"][idx]),
            process = "noteseq",
            dataset_name = self.dataset_name,
            fraction = 24,
            min_length = 16).get_tensor()
        data = self.transform(data)[0] #required unsqueeze
        label = "bla"
        return (data, label)

class ARNNDataset:
    def __init__(self, path: str, metadata: pd.DataFrame, dataset_name: str, split: str, mode: str, repeat=1):
        metadata = pd.concat([metadata]*repeat, axis=0)
        self.metadata = metadata.reset_index(drop=True)
        self.path = path
        self.model_name = "ARNN"
        self.dataset_name = dataset_name
        self.split = split
        self.transform = Compose(
                [
                    TensorRepresentation(filter_instruments=None),
                    RandomInstrument(keep_instruments="all") if self.split=="train" else DeterministicInstrument(),
                    RandomCrop(ctxt_size=16, fraction=24, model_name=self.model_name) if self.split=="train" else DeterministicCrop(ctxt_size=16, fraction=24, model_name=self.model_name), 
                ]
            )
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self,idx):
        data = EncodedMidi.from_path(
            file = os.path.join(self.path, self.metadata["filename"][idx]),
            process = "noteseq",
            dataset_name = self.dataset_name,
            fraction = 24,
            min_length = 16).get_tensor()
        data = self.transform(data) #required unsqueeze
        label = "bla"
        return (data, label)


class DeepBachDataset:
    def __init__(self, path: str, metadata: pd.DataFrame, dataset_name: str, repeat=1):
        metadata = pd.concat([metadata]*repeat, axis=0)
        self.metadata = metadata.reset_index(drop=True)
        self.path = path
        self.model_name = "DEEPBACH"
        self.dataset_name = dataset_name
        self.transform = Compose(
                [
                    #TensorRepresentation(filter_instruments=None),
                    #RandomInstrument(keep_instruments="all"),
                    RandomCrop(ctxt_size=16, fraction=16, model_name=self.model_name),
                    #RandomTranspose(bounds=(0,127)), #TODO: bounds por min/max en el dataset (habria que construirlo mientras se hacen los frames) 
                ]
            )
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self,idx):
        filepath = os.path.join(self.path, self.metadata["filename"][idx])
        #filepath = filepath if self.dataset_name != "jsb_chorales" else filepath.replace("mid", "mxl")
        data, metadata = EncodedMidi.from_path(
            file = filepath,
            process = "mxl_noteseq",
            dataset_name = self.dataset_name,
            fraction = 16,
            min_length = 16).get_tensor()
        mixed_data = torch.cat([data.unsqueeze(2), metadata], dim=2) #little hack to crop the same indexes in both data and metadata
        mixed_data = self.transform(mixed_data)
        data, metadata = mixed_data[:,:,0], mixed_data[:,:,1:]
        label = "BLA"
        return ([data, metadata], label)