from src.features.encoded_midi import EncodedMidi

from pathlib import Path
import os 
import numpy as np


project_dir = Path(__file__).resolve().parents[2]

data_dir = os.path.join(project_dir, "data")

ailabs_data = os.path.join(data_dir, "raw", "ailabs")
folk_data = os.path.join(data_dir, "raw", "folk")
jsb_chorales_data = os.path.join(data_dir, "raw", "jsb_chorales")

example_path = os.path.join(project_dir, "src", "models", "variable_length_piano_infilling", "model_src", "worded_data.pickle")
true_processed_data = np.load(example_path, allow_pickle=True)[0]

#true_data_path = os.path.join(ailabs_data, "0.mid")
true_data_path = os.path.join(folk_data, "sessiontune1.mid")

data = EncodedMidi.from_path(
            file = true_data_path,
            process = "remi",
            dataset_name = "remi",
            fraction = 16,
            min_length = 16).get_tensor()


def test_len():
	print(data)
	assert len(true_processed_data) == len(data)

def test_content():
	assert true_processed_data == data
