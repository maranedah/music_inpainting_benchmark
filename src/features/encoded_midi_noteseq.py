import argparse
import os 
import pretty_midi
import random
import numpy as np
from pathlib import Path



#PROJECT_DIR = Path(__file__).resolve().parents[2]
#FRAMES_DIR = os.path.join(PROJECT_DIR, "data", "frames")
#PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")
#DEFAULT_FRACTION = 24

class EncodedMidiNoteSeq:
	def __init__(self, path, fraction, dataset_name, min_length):
		self.path = path
		self.min_length = min_length
		self.fraction = fraction
		self.dataset_name = dataset_name

	def get_tensor(self):
		midi, time_grid = quantize(self.path, self.fraction, self.min_length)
		normalize_instrument_names(midi)
		note_seq = get_all_note_seq(midi, time_grid)
		names_inst = [x.name for x in midi.instruments]
		
		note_seqs = []
		for inst in note_seq:
			#note_seqs.append({"name": f"{self.path[:-4].split('/')[-1]}_{inst['inst_name']}"  , "notes":inst["notes"]})
			note_seqs.append({"name": f"{inst['inst_name']}"  , "notes":inst["notes"]})

		return note_seqs





# Quantizar archivos y guardar en "data/clean/DEFAULT_FRACTION". Incluir 
def get_max_time(midi):
	notes_per_instrument = [x.notes for x in midi.instruments]
	max_time = max([max(x, key=lambda x: x.end) for x in notes_per_instrument], key=lambda x: x.end).end
	return max_time

def get_grid(midi, max_time, DEFAULT_FRACTION, min_length):
	bpm_st, bpm_value = midi.get_tempo_changes()  
	bpm_st = np.append(bpm_st, max_time)
	#print(bpm_st, bpm_value)

	time_grid = []
	for i, (tempo_start, tempo_end) in enumerate(zip(bpm_st[:-1], bpm_st[1:])):

		bpm = int(bpm_value[i])
		TIME_PER_BAR = 60/bpm
		TIME_PER_MEASURE = TIME_PER_BAR*4   # 4/4 measures
		MIN_TIME_STEP = TIME_PER_MEASURE/DEFAULT_FRACTION

		current_time = tempo_start
		while current_time<tempo_end:
			time_grid.append(current_time)
			current_time+=MIN_TIME_STEP

	
	while len(time_grid)//24 < min_length:
		time_grid.append(current_time+MIN_TIME_STEP)
	
	return time_grid

def set_closest_time(time_grid, midi):
	notes_per_instrument = [x.notes for x in midi.instruments]
	
	for notes in notes_per_instrument:
		for note in notes:
			st_index = np.argmin(abs(time_grid - note.start))
			shift = time_grid[st_index] - note.start
			note.start += shift
			
			note.end += shift
			et_index = np.argmin(abs(time_grid - note.end))
			shift = time_grid[et_index] - note.end
			note.end += shift

			


def quantize(midi_file, fraction, min_length):
	midi = pretty_midi.PrettyMIDI(midi_file)
	#print(midi.instruments[0].notes)
	max_time = get_max_time(midi)
	midi.max_time = max_time
	time_grid = get_grid(midi, max_time, fraction, min_length)
	set_closest_time(time_grid, midi)
	return midi, time_grid
	
def get_note_seq(notes, time_grid, max_index):
	note_seq = [129]*max_index
	for note in notes:
		st_index = np.argmin(abs(time_grid - note.start))
		et_index = np.argmin(abs(time_grid - note.end))
		for i in range(st_index, et_index):
			note_seq[i] = 128
		note_seq[st_index] = note.pitch
	return note_seq 


def get_all_note_seq(midi, time_grid):
	max_index = len(time_grid)
	notes_per_instrument = [x.notes for x in midi.instruments]
	note_seqs = []
	for i, notes in enumerate(notes_per_instrument):
		note_seq = get_note_seq(notes, time_grid, max_index)
		inst_name = midi.instruments[i].name
		note_seqs.append({"inst_name":inst_name.lower(), "notes":note_seq})

	return note_seqs

def normalize_instrument_names(midi):
	for inst in midi.instruments:
		new_name = "".join([character for character in inst.name if character.isalnum()])
		inst.name = new_name.lower()
	return midi


if __name__ == "__main__":
	d = EncodedMidiNoteSeq(
		path = "/home/maraneda/music_inpainting_benchmark/data/raw/jsb_chorales/bwv122.6.mid",
		min_length = 16,
		fraction = 24,
		dataset_name = "jsb_chorales"
	).get_tensor()
	print(len(d[0]["notes"]))
	breakpoint()