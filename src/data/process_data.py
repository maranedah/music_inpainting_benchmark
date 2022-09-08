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

class Note():
    def __init__(self, row, ticks_per_note):
        self.track = row["track"]
        self.measure = row["measure"]
        self.rel_pos = round(row["relative_position"] * ticks_per_note)
        self.duration = round(row["duration"] * ticks_per_note)
        self.pitch = row["pitch"]
        self.velocity = row["velocity"]
        self.tempo = row["tempo"]

    def to_noteseq(self):
        return [self.pitch] + [128]*(self.duration-1)

    def to_vli(self):
        return [self.measure, self.relative_position, self.duration, self.pitch, self.velocity, self.tempo]

def to_noteseq(df, ticks_per_note):
    seqs = []
    overflow = []
    for keys, indexs in df.groupby(by=["track", "measure"]).groups.items():
        seq = [129 for _ in range(ticks_per_note*4)]
        for note in  df.loc[indexs,].iterrows():
            n = Note(note[1], ticks_per_note)
            noteseq = n.to_noteseq()
            # if noteseq se paso pal compas del lado?
            i = 0
            while len(overflow)>0:
                if i >= ticks_per_note*4:
                    break
                else:    
                    seq[i] = overflow.pop(0)
                    i+=1
            for i, ix in enumerate(range(n.rel_pos, n.rel_pos+n.duration)):
                if ix >= ticks_per_note*4:
                    overflow.append(noteseq[i])
                else:
                    seq[ix] = noteseq[i]

        seqs.append(seq)
    return seqs



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
            noteseq = to_noteseq(df, 6)
            breakpoint()
            



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()