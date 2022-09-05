import music21
import pretty_midi as pyd
import muspy
import os
import logging
import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = os.path.join(PROJECT_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
INTERIM_DIR = os.path.join(DATA_DIR, "interim")

def mxl_to_midi():
    bach_mxl = os.path.join(EXTERNAL_DIR, "jsb_chorales")
    out = os.path.join(RAW_DIR, "jsb_chorales")
    if not os.path.exists(out): os.mkdir(out) 
    
    for i, file in enumerate(os.listdir(bach_mxl)):
        print(f"Converting file {i}/{len(os.listdir(bach_mxl))}")
        path = os.path.join(bach_mxl, file)
        out_path = os.path.join(out, file[:-4] + ".mid")
        
        d = music21.converter.parse(path)
        d.write('midi', fp=out_path)

def cuantize():
    pass

def clean_repeated():
    pass


def main():
    logging.info('Creating necessary folders...')
    data_sources_file = open(os.path.join(RAW_DIR, "data_sources.json"), "r")
    sources = json.load(data_sources_file)
    for dataset, config in sources.items():
        raw_files = os.path.join(RAW_DIR, dataset)
        for file in os.listdir(raw_files):
            path = os.path.join(raw_files, file)
            if ".mxl" in file:
                data = muspy.read_musicxml(path)
            if ".mid" in file:
                data = muspy.read_midi(path)
            breakpoint()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()