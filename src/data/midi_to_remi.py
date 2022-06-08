import pretty_midi
import numpy as np
import os

import logging
from pathlib import Path

def load_song(midi_path):
	midi = pretty_midi.PrettyMIDI(midi_path)
	bpm_st, bpm_value = midi.get_tempo_changes()  
	#time_signatures = midi.time_signature_changes #asumamos 4/4 por ahora

	notes_per_instrument = [x.notes for x in midi.instruments]
	max_time = max([max(x, key=lambda x: x.end) for x in notes_per_instrument], key=lambda x: x.end).end
	bpm_st = np.append(bpm_st, max_time)

	#print(bpm_st, bpm_value)

	# Construct measures
	DEFAULT_FRACTION = 24
	
	measures = []
	for i in range(len(bpm_value)):
		bar_st = bpm_st[i]
		bpm = bpm_value[i]

		TIME_PER_BAR = 60/bpm
		TIME_PER_MEASURE = TIME_PER_BAR*4   # 4/4 measures
		MIN_TIME_STEP = TIME_PER_MEASURE/DEFAULT_FRACTION
	
		while bar_st < bpm_st[i+1]: #mientras no hayas construido compases que lleguen al cambio de tempo o al max_time
			bar_et =  bar_st + TIME_PER_MEASURE
			grid = np.linspace(bar_st, bar_et, DEFAULT_FRACTION+1)
			measure = {
				"grid": grid,
				"tempo": int(bpm),
				"duration_grid": np.arange(1,DEFAULT_FRACTION+1)*MIN_TIME_STEP,
			}

			measures.append(measure)
			bar_st = bar_et
	

	#print(measures)
	full_grid = np.unique(np.array([m["grid"] for m in measures])).flatten()
	#print(full_grid)

	# Quantizar
	# tomar cada nota y restarlo con la grid que armé. Asignarle el note.start de acuerdo a la resta más chica de la nota vs grid. 
	for notes in notes_per_instrument:
		for note in notes:
			index = np.argmin(abs(full_grid - note.start))
			shift = full_grid[index] - note.start
			note.start += shift
			#print(note.start, shift)
			
			note.end += shift
			index = np.argmin(abs(full_grid - note.end))
			shift = full_grid[index] - note.end
			note.end += shift


	# Agrupar por measure
	instruments_measures = []
	for notes in notes_per_instrument:
		notes.sort(key=lambda x: x.start)
		notes = iter(notes)
		note = next(notes, -1)
		for m in measures:
			notes_in_measures = []
			while note != -1 and note.start >= m["grid"][0] and note.start < m["grid"][-1]:
				#print(note)
				notes_in_measures.append(note)
				note = next(notes, -1)
			m["notes"] = notes_in_measures
		instruments_measures.append(measures)
	#print(len(instruments_measures[0]), len(grids_per_measure))
	#print(instruments_measures[0][0])


	# Transformar cada nota a representacion VLI
	for instrument in instruments_measures:
		for measure in instrument:
			measure["notes_remi"] = []
			for note in measure["notes"]:
				# ['Tempo', 'Bar', 'Position', 'Pitch', 'Duration', 'Velocity']
	
				tempo = measure["tempo"]
				position = np.argmin(abs(measure["grid"] - note.start))
				pitch = int(note.pitch)
				duration = np.argmin(abs(measure["duration_grid"] - (note.end - note.start)))
				velocity = note.velocity
				measure["notes_remi"].append([tempo, -1, position, pitch, duration, velocity])

	notes = [x["notes_remi"] for x in instruments_measures[0]]
	#notes = [x["notes_remi"] for instrument in instruments_measures for x in instrument]
	#[print(x) for x in notes[1]]
	return notes

def measure2noteseq(x):
	#print(x)
	a = [129]*24
	for note in x:
		a[note[2]]  = note[3]
		for j in range(note[4]):
			a[note[2]+j+1] = 128
	if tuple(a) == tuple([129]*24):
		print("compas vacio")
		return None
	return a




def main():
	project_dir = Path(__file__).resolve().parents[2]
	data_dir = os.path.join(project_dir, "data", "test")
	dataset_name = ""
	save_name = "test"
	midi_folder = os.path.join(data_dir, "raw", dataset_name)
	save_path = os.path.join(data_dir, "processed", "remi", dataset_name, save_name)

	if not os.path.exists(os.path.join(data_dir, "raw")): os.mkdir(os.path.join(data_dir, "raw")) 
	if not os.path.exists(os.path.join(data_dir, "processed")): os.mkdir(os.path.join(data_dir, "processed")) 
	if not os.path.exists(os.path.join(data_dir, "processed", "remi")): os.mkdir(os.path.join(data_dir, "processed", "remi")) 


	files = os.listdir(midi_folder)

	songs = []
	for file in files:
		midi = pretty_midi.PrettyMIDI(os.path.join(midi_folder, file))
		tsc = midi.time_signature_changes

		#print(tsc, len(tsc), tsc[0].numerator, tsc[0].denominator)
		if len(tsc) == 1 and tsc[0].numerator == 4 and tsc[0].denominator==4:
			d = load_song(os.path.join(midi_folder, file))
			songs.append(d)
			#print(d[1])
			#note_seq = [measure2noteseq(measure) for measure in d[1]]
			
			#a = np.array(note_seq).flatten()
			#if None not in a and len(a)>=24*16:
			#	note_seqs.append(a)

	#print(len(note_seqs))
	#note_seqs = np.array(note_seqs)
	np.save(save_path, songs)

if __name__ == '__main__':
	main()