from pathlib import Path
import os
from os.path import join

import pretty_midi
import shutil
import pandas as pd
import numpy as np
from itertools import groupby
import hashlib

import warnings
warnings.filterwarnings("ignore")


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = join(PROJECT_DIR, "data")
RAW_DIR = join(DATA_DIR, "raw")
FRAMES_DIR = join(DATA_DIR, "frames")

if not os.path.exists(FRAMES_DIR): os.mkdir(FRAMES_DIR) 

from src.features.encoded_midi import EncodedMidi
from src.features.transforms.MixMultiInstrument import MixMultiInstrument
from src.features.transforms.Compose import Compose
from src.features.transforms.TensorRepresentation import TensorRepresentation

def count_n_measures(path, filename):
	data = EncodedMidi.from_path(
            file = path,
            process = "remi",
            dataset_name = None,
            fraction = 16,
            min_length = 16).get_tensor()
	
	transform = Compose(
                [
					TensorRepresentation(filter_instruments=None),
                    MixMultiInstrument()
                ]
            )
	data = transform(data)
	n_measures = len(data[0])
	
	return n_measures

def count_measures(midi, max_time):
	
	bpm_st, bpm_value = midi.get_tempo_changes()  
	bpm_st = np.append(bpm_st, max_time)

	n_measures = 0
	for i in range(len(bpm_value)):
		bar_st = bpm_st[i]
		bpm = bpm_value[i]

		TIME_PER_BAR = 60/bpm
		TIME_PER_MEASURE = TIME_PER_BAR*4   # 4/4 measures

		while bar_st < bpm_st[i+1]: #mientras no hayas construido compases que lleguen al cambio de tempo o al max_time
			bar_et =  bar_st + TIME_PER_MEASURE
			n_measures+=1
			bar_st = bar_et
	return n_measures

def poly_vals(instruments):
	check = []
	max_polys = []
	polys = []

	for i, inst in enumerate(instruments):
		check.append(True)
		max_poly = 0
		poly = []		
		for key, group in groupby(inst.notes, lambda x: x.start):
			note_group = [x for x in group]
			n_poly = len(note_group)
			if n_poly > 1:
				check[i] = False
			if n_poly > max_poly:
				max_poly = n_poly
			poly.append(n_poly>1)
		max_polys.append(max_poly)
		polys.append(sum(poly)/len(poly))

	return check, max_polys, polys

def get_hash(file_path):
	md5_hash = hashlib.md5()
	a_file = open(file_path, "rb")
	content = a_file.read()
	md5_hash.update(content)
	digest = md5_hash.hexdigest()
	return digest 

def make_frames():
	folders = [x for x in os.listdir(RAW_DIR) if os.path.isdir(join(RAW_DIR, x))]
	for folder in folders:
		raw_folder = join(RAW_DIR, folder)
		
		column_names = ["filename", "n_instruments", "instrument_names", "n_notes", "is_monophony", "max_poly", "polys_percent", "tempo_changes", "tsc_length", "first_tsc", "tsc", "duration", "n_measures", "is_empty", "hash"]
		df = pd.DataFrame(columns = column_names)
		

		for i, file in enumerate(os.listdir(raw_folder)):
			midi_path = join(raw_folder, file)
			midi = pretty_midi.PrettyMIDI(midi_path)
			is_empty = int(len(midi.instruments)==0 or len(midi.instruments[0].notes) == 0)
			tsc = midi.time_signature_changes
			is_4_4 = len(tsc) == 1 and tsc[0].numerator == 4 and tsc[0].denominator == 4
			hash_val = get_hash(midi_path)

				
			if is_empty:
				d = {
					"filename":file,
					"n_instruments": 0,
					"instrument_names": -1,
					"n_notes": 0,
					"is_monophony": -1,
					"max_poly": 0,
					"polys_percent": 0,
					"tempo_changes": midi.get_tempo_changes(),
					"tsc_length": len(tsc),
					"first_tsc": f"{tsc[0].numerator}/{tsc[0].denominator}",
					"is_4_4": int(is_4_4),
					"tsc": tsc,
					"duration": 0,
					"n_measures": 0,
					"is_empty": is_empty,
					"hash": hash_val,
					
				}
			else:
				notes_per_instrument = [x.notes for x in midi.instruments]
				max_time = max([max(x, key=lambda x: x.end) for x in notes_per_instrument], key=lambda x: x.end).end
				#n_measures = count_n_measures(midi_path, file)
				n_measures = count_measures(midi, max_time) #if is_4_4 else -1
				is_monophony, max_poly, polys_percent = poly_vals(midi.instruments)
				
				d = {
					"filename":file,
					"n_instruments": len(midi.instruments),
					"instrument_names": tuple([x.name for x in midi.instruments]),
					"n_notes": tuple([len(x) for x in notes_per_instrument]),
					"is_monophony": tuple(is_monophony),
					"max_poly": tuple(max_poly),
					"polys_percent": tuple(polys_percent),
					"tempo_changes": midi.get_tempo_changes(),
					"tsc_length": len(tsc),
					"first_tsc": f"{tsc[0].numerator}/{tsc[0].denominator}",
					"is_4_4": int(is_4_4),
					"tsc": tsc,#[f"{x.numerator}/{x.denominator}" for x in tsc],
					"duration": max_time,
					"n_measures": n_measures,
					"is_empty": is_empty,
					"hash": hash_val,
					
				}

			df = df.append(d, ignore_index=True)
			if i%100==0: 
				print(f"{i}/{len(os.listdir(raw_folder))}")
				
		#print(df)
		df.to_pickle(join(FRAMES_DIR, folder+".pkl"))


