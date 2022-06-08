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
from .transforms.Squeeze import Squeeze

class MidiDataset:
    def __init__(
            self, 
            representation: str,
            path: str, 
            metadata: pd.DataFrame, 
            dataset_name: str,
            fraction: int,
            ctxt_size: int,
            transform,
            repeat=1
        ):
        self.metadata = pd.concat([metadata]*repeat, axis=0).reset_index(drop=True)
        self.path = path
        self.dataset_name = dataset_name
        self.transform = transform 
        self.representation = representation
        self.fraction = fraction
        self.ctxt_size = ctxt_size
        self.cache = {}

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self,idx):
        if not self.metadata["filename"][idx] in self.cache:
            data = EncodedMidi.from_path(
                file = os.path.join(self.path, self.metadata["filename"][idx]),
                process = self.representation,
                dataset_name = self.dataset_name,
                fraction = self.fraction,
                min_length = self.ctxt_size
            ).get_tensor()
            self.cache[self.metadata["filename"][idx]] = data
        else:
            data = self.cache[self.metadata["filename"][idx]]
        data = self.transform(data)
        return (data, "bla")

def get_dataset(model_name: str, dataset_name: str, df: pd.DataFrame, split: str, path: str, repeat: int, is_polyphony = False):
    is_train = split == "train"
    metadata = df[df["set"] == split]
        
    if model_name in ["vae_SKETCHNET", "vae_GRU_VAE"]:
        fraction = 24
        ctxt_size = 1
        representation = "noteseq"
        bounds = (0,127) #TODO: bounds por min/max en el dataset (habria que construirlo mientras se hacen los frames)
        transform = Compose(
            [
                TensorRepresentation(filter_instruments=None),
                RandomInstrument(is_train),
                RandomCrop(ctxt_size, fraction, model_name, is_train),
                RandomTranspose(bounds, representation, is_train),  
                Factorize((1,))
            ]
        )

    elif model_name in ["SKETCHNET", "GRU_VAE"]:
        fraction = 24
        ctxt_size = 16
        representation = "noteseq"
        bounds = (0,127) #TODO: bounds por min/max en el dataset (habria que construirlo mientras se hacen los frames)
        transform = Compose(
            [
                TensorRepresentation(filter_instruments=None),
                RandomInstrument(is_train),
                RandomCrop(ctxt_size, fraction, model_name, is_train),
                RandomTranspose(bounds, representation, is_train),  
                Factorize(ctxt_split=(6, 4, 6), split_size=fraction)
            ]
        )

    elif model_name == "VLI":
        fraction = 16
        split_size = 1
        ctxt_size = 16
        representation = "remi"
        bounds=(0,127)
        if is_train:
            transform = Compose(
                    [
                        TensorRepresentation(filter_instruments=None),
                        #MixMultiInstrument() if is_polyphony else Identity(),
                        #RandomInstrument(is_train) if not(is_polyphony) else Identity(),
                        RandomCrop(ctxt_size=16, fraction=1, model_name=model_name, is_train=is_train),  #fraction=1 es pq no hay split_size, cada measures un elemento del arreglo de data *buscar mejor nombre*
                        AssignBarNumbers(ctxt_size=16),
                        RandomTranspose(bounds=(0,127), representation="remi", is_train=is_train), #TODO: bounds por min/max en el dataset (habria que construirlo mientras se hacen los frames) 
                        PadWords(fraction=16),
                        
                    ]
                )
        else:
            transform = Compose(
                    [
                        TensorRepresentation(filter_instruments=None),
                        MixMultiInstrument() if is_polyphony else Identity(),
                        DeterministicInstrument() if not(is_polyphony) else Identity(),
                        DeterministicCrop(ctxt_size=16, fraction=1, model_name=model_name),  #fraction=1 es pq no hay split_size, cada measures un elemento del arreglo de data *buscar mejor nombre*
                        AssignBarNumbers(ctxt_size=16),
                        PadWords(fraction=16),
                    ]
                )            
            
    elif model_name == "vae_INPAINTNET":
        fraction = 24
        ctxt_size = 1
        representation = "noteseq"
        bounds = (0,127) #TODO: bounds por min/max en el dataset (habria que construirlo mientras se hacen los frames)
        transform = Compose(
                [
                    TensorRepresentation(filter_instruments=None),
                    RandomInstrument(is_train),
                    RandomCrop(ctxt_size, fraction, model_name, is_train),
                    Squeeze(dim=0) 
                ]
            )

    elif model_name == "INPAINTNET":
        fraction = 24
        ctxt_size = 16
        representation = "noteseq"
        bounds = (0,127) #TODO: bounds por min/max en el dataset (habria que construirlo mientras se hacen los frames)
        transform = Compose(
                [
                    TensorRepresentation(filter_instruments=None),
                    RandomInstrument(is_train),
                    RandomCrop(ctxt_size, fraction, model_name, is_train),
                    Squeeze(dim=0) 
                ]
            )

    elif model_name == "ARNN":
        fraction = 24
        ctxt_size = 16
        representation = "noteseq"
        bounds = (0,127) #TODO: bounds por min/max en el dataset (habria que construirlo mientras se hacen los frames)
        transform = Compose(
                [
                    TensorRepresentation(filter_instruments=None),
                    RandomInstrument(is_train),
                    RandomCrop(ctxt_size, fraction, model_name, is_train), 
                ]
            )

    elif model_name == "DEPBACH":
        fraction = 16
        ctxt_size = 16
        transform = Compose(
                [
                    RandomCrop(ctxt_size, fraction, model_name, is_train),
                ]
            )
        

    return MidiDataset(representation, path, metadata, dataset_name, fraction, ctxt_size, transform, repeat)
