import logging
from pathlib import Path
import requests
import os
import zipfile
import shutil
import gdown
import json
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = os.path.join(PROJECT_DIR, "data")
EXTERNAL_DIR = os.path.join(DATA_DIR, "external")
RAW_DIR = os.path.join(DATA_DIR, "raw")
INTERIM_DIR = os.path.join(DATA_DIR, "interim")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

def create_folders():
    if not os.path.exists(DATA_DIR): os.mkdir(DATA_DIR)
    if not os.path.exists(EXTERNAL_DIR): os.mkdir(EXTERNAL_DIR) 
    if not os.path.exists(RAW_DIR): os.mkdir(RAW_DIR) 
    if not os.path.exists(INTERIM_DIR): os.mkdir(INTERIM_DIR) 
    if not os.path.exists(PROCESSED_DIR): os.mkdir(PROCESSED_DIR) 

    
def download_files(dataset_name, url, path, file_format, compression_format, gdownload=False, subdirectories=True):
    OUTPUT_FILE = os.path.join(RAW_DIR, f"{dataset_name}{compression_format}")
    if not gdownload:
        r = requests.get(url)
        open(OUTPUT_FILE, 'wb').write(r.content)
    else:
        gdown.download(url, OUTPUT_FILE, quiet=False)

    OUTPUT_DIR = os.path.join(RAW_DIR, dataset_name)
    TEMP_DIR = os.path.join(RAW_DIR, "temp")

    if not os.path.exists(TEMP_DIR): os.mkdir(TEMP_DIR) 
    if not os.path.exists(OUTPUT_DIR): os.mkdir(OUTPUT_DIR) 

    shutil.unpack_archive(OUTPUT_FILE, extract_dir=TEMP_DIR)
    for root, subdirs, files in tqdm(os.walk(TEMP_DIR)):
        if subdirectories and len(subdirs)==0: 
            [
                shutil.move(
                    os.path.join(root, file), 
                    os.path.join(OUTPUT_DIR, file)
                )
                for file in files 
                if path in root and file_format in file
            ]
    shutil.rmtree(TEMP_DIR)    
    os.remove(OUTPUT_FILE)


def main():
    logging.info('Creating necessary folders...')
    create_folders()
    data_sources_file = open(os.path.join(RAW_DIR, "data_sources.json"), "r")
    sources = json.load(data_sources_file)
    for dataset, config in sources.items():
        download_files(dataset, **config)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()