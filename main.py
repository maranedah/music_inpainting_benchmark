import yaml
from src.features.dataset import Dataset

file = open('hiperparameters.yaml')
hiperparameters = yaml.load(file, Loader=yaml.FullLoader)

model = "sketch_vae"
dataset = "IrishFolkSong"
data_type = "measures"
split = "train"
format_ = "noteseq"
resolution = 24
transformations = hiperparameters[model]["data_transformations"]

train_set = Dataset(dataset, data_type, split, format_, resolution, transformations)
train_set[0]
breakpoint()
