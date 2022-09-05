# Transforms interim into vectors
# FORCE 4/4?
# - NoteSeq
#   - 16
#   - 24
#   - 32
#   - 48
# - REMI
#   - 16
#   - 24
#   - 32
#   - 48
# - VAEVectors
#   - 16
#   - 24
#   - 32
#   - 48 

import music21
import pretty_midi as pyd
import muspy
import pandas as pd
import os
import logging
import json
import numpy
from pathlib import Path
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = os.path.join(PROJECT_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
INTERIM_DIR = os.path.join(DATA_DIR, "interim")


def to_noteseq(df):
    pass


def main():
    logging.info('Creating necessary folders...')
    data_sources_file = open(os.path.join(RAW_DIR, "data_sources.json"), "r")
    sources = json.load(data_sources_file)
    for dataset, config in sources.items():
        interim_files = os.path.join(INTERIM_DIR, dataset)
        out_dir = os.path.join(PROCESSED_DIR, dataset)
        if not os.path.exists(out_dir): os.mkdir(out_dir)
        for file in tqdm(os.listdir(interim_files)):
            path = os.path.join(interim_files, file) 
            df = pd.read_csv(path)
            to_noteseq(df)
            breakpoint()



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()