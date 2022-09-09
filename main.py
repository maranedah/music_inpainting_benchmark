import yaml
import os
import torch
from src.features.dataset import Dataset

file = open('hiperparameters.yaml')
hiperparameters = yaml.load(file, Loader=yaml.FullLoader)

batch_size = 8
model = "inpaintnet"
dataset = "IrishFolkSong"
resolution = 24
data_type = hiperparameters[model]["data_type"]
format_ = hiperparameters[model]["format"]
transformations = hiperparameters[model]["data_transformations"]
split = "train"

train_set, val_set, test_set = [
    Dataset(
        dataset, 
        data_type, 
        split, 
        format_, 
        resolution, 
        transformations
    )
    for split in ["train", "val", "test"]
]

train_loader, val_loader, test_loader = [
    torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=i==0,
        num_workers=os.cpu_count(),
        pin_memory=True,
        drop_last=True,
    )
    for i, dataset in enumerate([train_set, val_set, test_set])
]


breakpoint()
