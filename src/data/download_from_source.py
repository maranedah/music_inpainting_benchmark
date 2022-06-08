# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import requests
import music21

import os
import zipfile
import shutil
import gdown


BACH_CHORALES = "https://github.com/cuthbertLab/music21/archive/refs/heads/master.zip"
FOLK = "https://github.com/IraKorshunova/folk-rnn/raw/master/data/midi.tgz"
AILABS = "https://drive.google.com/file/d/1qw_tVUntblIg4lW16vbpjLXVndkVtgDe/view"


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = os.path.join(PROJECT_DIR, "data")
EXTERNAL_DIR = os.path.join(PROJECT_DIR, "data", "external")
RAW_DIR = os.path.join(PROJECT_DIR, "data", "raw")
INTERIM_DIR = os.path.join(PROJECT_DIR, "data", "interim")
PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")
FRAMES_DIR = os.path.join(PROJECT_DIR, "data", "frames")




def download_bach_files():
    r = requests.get(BACH_CHORALES)
    print(r)

    OUTPUT_FILE = os.path.join(EXTERNAL_DIR, "jsb_chorales.zip")
    OUTPUT_DIR = os.path.join(EXTERNAL_DIR, "jsb_chorales")
    TEMP_DIR = os.path.join(EXTERNAL_DIR,"temp")

    open(OUTPUT_FILE, 'wb').write(r.content)
    if not os.path.exists(TEMP_DIR): os.mkdir(TEMP_DIR) 
    if not os.path.exists(OUTPUT_DIR): os.mkdir(OUTPUT_DIR) 
    
    with zipfile.ZipFile(OUTPUT_FILE, 'r') as zip_ref:
        listOfFileNames = zip_ref.namelist()
        files = list(filter(lambda x: x.startswith("music21-master/music21/corpus/bach") and x.endswith(".mxl"), listOfFileNames))
        [zip_ref.extract(file, TEMP_DIR) for file in files]


    for root, subdirectories, files in os.walk(TEMP_DIR):
        for file in files:
            shutil.move(os.path.join(root, file), os.path.join(OUTPUT_DIR, file))

    shutil.rmtree(TEMP_DIR)    
    os.remove(OUTPUT_FILE)

def download_folk_files():
    r = requests.get(FOLK)

    OUTPUT_FILE = os.path.join(RAW_DIR, "folk.tgz")
    OUTPUT_DIR = os.path.join(RAW_DIR, "folk")
    TEMP_DIR = os.path.join(RAW_DIR,"temp")

    open(OUTPUT_FILE, 'wb').write(r.content)
    if not os.path.exists(TEMP_DIR): os.mkdir(TEMP_DIR) 
    if not os.path.exists(OUTPUT_DIR): os.mkdir(OUTPUT_DIR) 
    
    shutil.unpack_archive(OUTPUT_FILE, extract_dir=TEMP_DIR)
    for root, subdirectories, files in os.walk(TEMP_DIR):
        for file in files:
            if file.endswith(".mid") or file.endswith(".midi"):
                shutil.move(os.path.join(root, file), os.path.join(OUTPUT_DIR, file))

    shutil.rmtree(TEMP_DIR)    
    os.remove(OUTPUT_FILE)

def download_ailabs_files():

    url = "https://drive.google.com/uc?id=1qw_tVUntblIg4lW16vbpjLXVndkVtgDe"
    OUTPUT_FILE = os.path.join(RAW_DIR, "ailabs.zip")
    OUTPUT_DIR = os.path.join(RAW_DIR, "ailabs")
    TEMP_DIR = os.path.join(RAW_DIR,"temp")
    gdown.download(url, OUTPUT_FILE, quiet=False)
    
    if not os.path.exists(TEMP_DIR): os.mkdir(TEMP_DIR) 
    if not os.path.exists(OUTPUT_DIR): os.mkdir(OUTPUT_DIR) 
    
    shutil.unpack_archive(OUTPUT_FILE, extract_dir=TEMP_DIR)

    for root, subdirectories, files in os.walk(TEMP_DIR):
        for file in files:
            if "midi_synchronized" in root and file[0]!= "." and (file.endswith(".mid") or file.endswith(".midi")):
                shutil.move(os.path.join(root, file), os.path.join(OUTPUT_DIR, file))

    shutil.rmtree(TEMP_DIR)    
    os.remove(OUTPUT_FILE)


import subprocess
def bach_to_midi():
    bach_mxl = os.path.join(EXTERNAL_DIR, "jsb_chorales")
    out = os.path.join(RAW_DIR, "jsb_chorales")
    if not os.path.exists(out): os.mkdir(out) 
    
    for i, file in enumerate(os.listdir(bach_mxl)):
        print(f"Converting file {i}/{len(os.listdir(bach_mxl))}")
        path = os.path.join(bach_mxl, file)
        out_path = os.path.join(out, file[:-4] + ".mid")
        
        d = music21.converter.parse(path)
        d.write('midi', fp=out_path)
    
def create_folders():
    if not os.path.exists(DATA_DIR): os.mkdir(DATA_DIR)
    if not os.path.exists(EXTERNAL_DIR): os.mkdir(EXTERNAL_DIR) 
    if not os.path.exists(RAW_DIR): os.mkdir(RAW_DIR) 
    if not os.path.exists(INTERIM_DIR): os.mkdir(INTERIM_DIR) 
    if not os.path.exists(PROCESSED_DIR): os.mkdir(PROCESSED_DIR) 
    if not os.path.exists(FRAMES_DIR): os.mkdir(FRAMES_DIR) 

def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Collecting datasets from sources')
    
    create_folders()
    download_folk_files()
    download_ailabs_files()
    download_bach_files()
    bach_to_midi()
    




if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    load_dotenv(find_dotenv())

    main()
