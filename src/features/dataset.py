import pandas as pd
import os 
import numpy as np
from .transforms import *
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = os.path.join(PROJECT_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
SPLIT_DIR = os.path.join(DATA_DIR, "splits")




class Dataset:
    def __init__(self, dataset, data_type, split, format_, resolution, transformations):
        path = os.path.join(SPLIT_DIR, f"{dataset}_{data_type}.csv")
        df = pd.read_csv(path)
        self.df = df[df["set"] == split]
        self.dataset = dataset
        self.data_type = data_type
        self.format = format_
        self.resolution = resolution
        self.cache = {}
        self.transformations = Compose(
            [globals()[tf["name"]](*tf["params"], resolution=resolution) for tf in transformations]
        )
        
    def __len__(self):
        return len(df)

    def __getitem__(self, idx):
        filename = self.df["filename"][idx]
        track = self.df["track"][idx]
        

        file_path = os.path.join(PROCESSED_DIR, self.dataset, f"{self.format}_{self.resolution}", filename)
        if not filename in self.cache:

            data = np.load(file_path, allow_pickle=True)
            if self.data_type == "measures":
                measure = self.df["measure"][idx]
                data = data[track][measure]
            elif self.data_type == "contexts":
                st_index = self.df["measure_st"][idx]
                end_index = self.df["measure_end"][idx]
                data = data[track][st_index:end_index]

            data = data.astype(int)
            data = self.transformations(data)
            self.cache[filename] = data
        else:
            data = self.cache[filename]
        return data


