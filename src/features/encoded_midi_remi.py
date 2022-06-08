import pretty_midi
import numpy as np
import os
import torch
import miditoolkit
import collections

import logging
from pathlib import Path

# parameters for input
DEFAULT_VELOCITY_BINS = np.linspace(0, 128, 32+1, dtype=np.int)
DEFAULT_FRACTION = 16
DEFAULT_DURATION_BINS = np.arange(60, 3841, 60, dtype=int)
DEFAULT_TEMPO_INTERVALS = [range(30, 90), range(90, 150), range(150, 210)]

# parameters for output
DEFAULT_RESOLUTION = 480

tempo_quantize_step = 4

GroupEvent = collections.namedtuple('GroupEvent', ['Tempo', 'Bar', 'Position', 'Pitch', 'Duration', 'Velocity'])

class Item(object):
    def __init__(self, name, start, end, velocity, pitch):
        self.name = name
        self.start = start
        self.end = end
        self.velocity = velocity
        self.pitch = pitch

    def __repr__(self):
        return 'Item(name={}, start={}, end={}, velocity={}, pitch={})'.format(
            self.name, self.start, self.end, self.velocity, self.pitch)

class EncodedMidiRemi:
	def __init__(self, path, fraction, dataset_name, min_length):
		self.path = path
		self.min_length = min_length
		self.fraction = fraction
		self.dataset_name = dataset_name
		self.ctxt_size = 16

	def get_tensor(self):
		e2w, w2e = construct_dict()
		inst_events, inst_names = extract_tuple_events(self.path)
		insts = []
		for events, name in zip(inst_events, inst_names):
			events = group_by_bar(events)
			words_in_midi = tuple_event_to_word(events, e2w)
			insts.append({"name": name, "notes": words_in_midi})
		return insts

	

###################################
def extract_tuple_events(input_path):
	inst_items, tempo_items, inst_names = read_items(input_path)
	#note_items = note_items[0] # assume there is only 1 track, so this get the first track
	inst_events = []
	for note_items in inst_items:
		note_items = quantize_items(note_items)
		max_time = note_items[-1].end
		items = tempo_items + note_items
		groups = group_items(items, max_time)
		events = item2event(groups)
		events = convert_to_tuple_events(events, tempo_items)
		inst_events.append(events)
	return inst_events, inst_names


def read_items(file_path):
	midi_obj = miditoolkit.midi.parser.MidiFile(file_path)
	# note
	all_note_items = [[] for _ in range(len(midi_obj.instruments))]
	for i, instrument in enumerate(midi_obj.instruments):
		notes = instrument.notes
		notes.sort(key=lambda x: (x.start, x.pitch))
		for note in notes:
			all_note_items[i].append(Item(
				name='Note',
				start=note.start,
				end=note.end,
				velocity=note.velocity,
				pitch=note.pitch))
		all_note_items[i].sort(key=lambda x: x.start)
	# tempo
	tempo_items = []
	for tempo in midi_obj.tempo_changes:
		tempo_items.append(Item(
			name='Tempo',
			start=tempo.time,
			end=None,
			velocity=None,
			pitch=int(tempo.tempo)))
	tempo_items.sort(key=lambda x: x.start)
	# expand to all beat
	max_tick = tempo_items[-1].start
	existing_ticks = {item.start: item.pitch for item in tempo_items}
	wanted_ticks = np.arange(0, max_tick+1, DEFAULT_RESOLUTION)
	output = []
	for tick in wanted_ticks:
		if tick in existing_ticks:
			output.append(Item(
				name='Tempo',
				start=tick,
				end=None,
				velocity=None,
				pitch=existing_ticks[tick]))
		else:
			output.append(Item(
				name='Tempo',
				start=tick,
				end=None,
				velocity=None,
				pitch=output[-1].pitch))
	tempo_items = output
	return all_note_items, tempo_items, [x.name.lower() for x in midi_obj.instruments]


# quantize items
def quantize_items(items, ticks=120):
	if len(items) == 1 and items[0].start == 0:
		return items
	# grid
	grids = np.arange(0, items[-1].start, ticks, dtype=int)
	# process
	for item in items:
		index = np.argmin(abs(grids - item.start))
		shift = grids[index] - item.start
		item.start += shift
		item.end += shift
	return items



# group items
def group_items(items, max_time, ticks_per_bar=DEFAULT_RESOLUTION*4):
	items.sort(key=lambda x: x.start)
	downbeats = np.arange(0, max_time+ticks_per_bar, ticks_per_bar)
	groups = []
	for db1, db2 in zip(downbeats[:-1], downbeats[1:]):
		insiders = []
		for item in items:
			if (item.start >= db1) and (item.start < db2):
				insiders.append(item)
		overall = [db1] + insiders + [db2]
		groups.append(overall)
	return groups

class Event(object):
    def __init__(self, name, time, value, text):
        self.name = name
        self.time = time
        self.value = value
        self.text = text

    def __repr__(self):
        return 'Event(name={}, time={}, value={}, text={})'.format(
            self.name, self.time, self.value, self.text)

def item2event(groups):
	events = []
	n_downbeat = 0
	for i in range(len(groups)):
		# if 'Note' not in [item.name for item in groups[i][1:-1]]:
		#     continue
		bar_st, bar_et = groups[i][0], groups[i][-1]
		n_downbeat += 1
		events.append(Event(
			name='Bar',
			time=None,
			value=None,
			text='{}'.format(n_downbeat)))
		for item in groups[i][1:-1]:
			# position
			flags = np.linspace(bar_st, bar_et, DEFAULT_FRACTION, endpoint=False)
			index = np.argmin(abs(flags-item.start))
			events.append(Event(
				name='Position',
				time=item.start,
				# value='{}/{}'.format(index+1, utils.DEFAULT_FRACTION),
				value='{}/{}'.format(index, DEFAULT_FRACTION),
				text='{}'.format(item.start)))
			if item.name == 'Note':
				# velocity
				velocity_index = np.searchsorted(
					DEFAULT_VELOCITY_BINS,
					item.velocity,
					side='right') - 1
				events.append(Event(
					name='Velocity',
					time=item.start,
					value=velocity_index,
					text='{}/{}'.format(item.velocity, DEFAULT_VELOCITY_BINS[velocity_index])))
				# pitch
				events.append(Event(
					name='Pitch',
					time=item.start,
					value=item.pitch,
					text='{}'.format(item.pitch)))
				# duration
				duration = item.end - item.start
				index = np.argmin(abs(DEFAULT_DURATION_BINS-duration))
				events.append(Event(
					name='Duration',
					time=item.start,
					value=index,
					text='{}/{}'.format(duration, DEFAULT_DURATION_BINS[index])))
			elif item.name == 'Chord':
				events.append(Event(
					name='Chord',
					time=item.start,
					value=item.pitch,
					text='{}'.format(item.pitch)))
			elif item.name == 'Tempo':
				tempo = item.pitch
				if tempo in DEFAULT_TEMPO_INTERVALS[0]:
					tempo_style = Event('Tempo Class', item.start, 'slow', None)
					tempo_value = Event('Tempo Value', item.start,
						tempo-DEFAULT_TEMPO_INTERVALS[0].start, None)
				elif tempo in DEFAULT_TEMPO_INTERVALS[1]:
					tempo_style = Event('Tempo Class', item.start, 'mid', None)
					tempo_value = Event('Tempo Value', item.start,
						tempo-DEFAULT_TEMPO_INTERVALS[1].start, None)
				elif tempo in DEFAULT_TEMPO_INTERVALS[2]:
					tempo_style = Event('Tempo Class', item.start, 'fast', None)
					tempo_value = Event('Tempo Value', item.start,
						tempo-DEFAULT_TEMPO_INTERVALS[2].start, None)
				elif tempo < DEFAULT_TEMPO_INTERVALS[0].start:
					tempo_style = Event('Tempo Class', item.start, 'slow', None)
					tempo_value = Event('Tempo Value', item.start, 0, None)
				elif tempo >= DEFAULT_TEMPO_INTERVALS[2].stop:
					tempo_style = Event('Tempo Class', item.start, 'fast', None)
					tempo_value = Event('Tempo Value', item.start, 59, None)
				events.append(tempo_style)
				events.append(tempo_value)
	return events


def convert_to_tuple_events(events, tempo_items):
	
	group_events = []
	group_event_template = {'Position': None, 'Pitch': None, 'Duration': None, 'Velocity': None}
	group_event = group_event_template.copy()
	bar_value = None
	tempo = 1
	tempo_class = None
	for i, event in enumerate(events):
		if event.name == 'Bar':
			bar_value = int(event.text)
		elif event.name == 'Tempo Value':
			tempo = event.value
		elif event.name == 'Tempo Class':
			tempo_class = event.value
		else:
			group_event[event.name] = event.value

		if None not in group_event.values():
			if tempo_class == 'slow':
				cur_tempo = DEFAULT_TEMPO_INTERVALS[0].start + tempo
			elif tempo_class == 'mid':
				cur_tempo = DEFAULT_TEMPO_INTERVALS[1].start + tempo
			elif tempo_class == 'fast':
				cur_tempo = DEFAULT_TEMPO_INTERVALS[2].start + tempo
			else:
				raise Exception("Undefined tempo class: %s" % tempo_class)
			group_event['Bar'] = bar_value
			group_event['Tempo'] = cur_tempo
			group_events.append(GroupEvent(**group_event))
			group_event = group_event_template.copy()

	return group_events


def group_by_bar(events):
	bar = None
	grouped_events = [] # Events grouped by bar [[events of bar0], [events of bar1], [events of bar2], ...]
	for e in events:
		if bar != e.Bar:
			bar = e.Bar
			grouped_events.append([])
		grouped_events[-1].append(e)

	return grouped_events

def tuple_event_to_word(data, e2w):

	words_in_midi = []
	for bar in data:
		words_in_bar = []
		for event in bar:
			words = [e2w['Tempo']['Tempo %d' % (event.Tempo - event.Tempo % tempo_quantize_step)],
					-1, # set value when the 8-bar chunk is selected (0 ~ 7), not e2w['Bar %d' % event.Bar]
					e2w['Position']['Position %s' % event.Position],
					e2w['Pitch']['Pitch %d' % event.Pitch],
					e2w['Duration']['Duration %d' % event.Duration],
					e2w['Velocity']['Velocity %d' % event.Velocity]]
			words_in_bar.append(words)
		words_in_midi.append(words_in_bar)
	
	return words_in_midi
























































def construct_dict():
	event2word = {}
	word2event = {}
	tempo_quantize_step = 4
	ctxt_size = 16
	velocity_bins = 32

	for etype in ['Tempo', 'Bar', 'Position', 'Pitch', 'Duration', 'Velocity']:
		count = 0
		e2w = {}

		# Tempo 30 ~ 210
		if etype == 'Tempo':
			for i in range(28, 211, tempo_quantize_step):
				e2w['Tempo %d' % i] = count
				count += 1

		# Bar 0 ~ 15
		elif etype == 'Bar':
			for i in range(ctxt_size):
				e2w['Bar %d' % i] = count
				count += 1

		# Position: 0/16 ~ 15/16
		elif etype == 'Position':
			for i in range(0, 16):
				e2w['Position %d/16' % i] = count
				count += 1

		# Pitch: 22 ~ 107
		elif etype == 'Pitch':
			for i in range(22, 108):
				e2w['Pitch %d' % i] = count
				count += 1

		# Duration: 0 ~ 63
		elif etype == 'Duration':
			for i in range(64):
				e2w['Duration %d' % i] = count
				count += 1

		# Velocity: 0 ~ 21
		elif etype == 'Velocity':
			for i in range(velocity_bins):
				e2w['Velocity %d' % i] = count
				count += 1

		else:
			raise Exception('etype error')


		e2w['%s <BOS>' % etype] = count
		count += 1
		e2w['%s <EOS>' % etype] = count
		count += 1
		e2w['%s <PAD>' % etype] = count
		count += 1

		event2word[etype] = e2w
		word2event[etype] = {e2w[key]: key for key in e2w}

	#print(event2word)
	return event2word, word2event


def categorize_ranges(notes, dictionary):
	TEMPO_BINS = np.arange(28, 211, 4, dtype=int)
	VELOCITY_BINS = np.arange(0, 128, 4, dtype=int)


	for inst in notes:
		for measure in inst["notes"]:
			for note in measure:
				note[0] = np.argmin(abs(note[0] - TEMPO_BINS)) # Tempo
				#note[1] = #Bar is set when the contexts are split
				#note[2] = #Position
				note[3] = dictionary[0]['Pitch'][f'Pitch {note[3]}']#Pitch
				#note[4] = #Duration
				note[5] = np.argmin(abs(note[5] - VELOCITY_BINS))#Velocity
	return notes
		
def quantize(notes_per_instrument, full_grid):
	for notes in notes_per_instrument:
		for note in notes:
			index = np.argmin(abs(full_grid - note.start))
			shift = full_grid[index] - note.start
			note.start += shift
			
			note.end += shift
			index = np.argmin(abs(full_grid - note.end))
			shift = full_grid[index] - note.end
			note.end += shift
	return notes_per_instrument
	
def get_measures(bpm_value, bpm_st, fraction):
	measures = []
	for i in range(len(bpm_value)):
		bar_st = bpm_st[i]
		bpm = bpm_value[i]

		TIME_PER_BAR = 60/bpm
		TIME_PER_MEASURE = TIME_PER_BAR*4   # 4/4 measures
		MIN_TIME_STEP = TIME_PER_MEASURE/fraction
	
		while bar_st < bpm_st[i+1]: #mientras no hayas construido compases que lleguen al cambio de tempo o al max_time
			bar_et =  bar_st + TIME_PER_MEASURE
			grid = np.linspace(bar_st, bar_et, fraction+1)
			measure = {
				"grid": grid,
				"tempo": int(bpm),
				"duration_grid": np.arange(1,fraction+1)*MIN_TIME_STEP,
			}

			measures.append(measure)
			bar_st = bar_et
	return measures

def get_time_grid(measures):
	full_grid = np.unique(np.array([m["grid"] for m in measures])).flatten()
	return full_grid

def group_by_measures(notes_per_instrument, measures):
	inst_measures = [[] for i in range(len(notes_per_instrument))]

	for i, notes in enumerate(notes_per_instrument):
		notes.sort(key=lambda x: x.start)
		notes = iter(notes)
		note = next(notes, -1)
		for m in measures:
			notes_in_measures = []
			while note != -1 and note.start >= m["grid"][0] and note.start < m["grid"][-1]:
				#print(note)
				notes_in_measures.append(note)
				note = next(notes, -1)
			inst_measures[i].append(notes_in_measures)
	return inst_measures


def notes_to_remi(instruments_measures, measures_data):
	inst_measures = [[] for i in range(len(instruments_measures))]
	
	for i, instrument in enumerate(instruments_measures):
		for j, group in enumerate(instrument):
			m = []
			for note in group:
				# ['Tempo', 'Bar', 'Position', 'Pitch', 'Duration', 'Velocity']
				tempo = measures_data[j]["tempo"]
				position = np.argmin(abs(measures_data[j]["grid"] - note.start))
				pitch = int(note.pitch)
				duration = np.argmin(abs(measures_data[j]["duration_grid"] - (note.end - note.start)))
				velocity = note.velocity
				m.append([tempo, -1, position, pitch, duration, velocity])
			inst_measures[i].append(m)
		
	
	return inst_measures

def normalize_instrument_names(midi):
	for inst in midi.instruments:
		new_name = "".join([character for character in inst.name if character.isalnum()])
		inst.name = new_name.lower()
	return midi


def load_song(midi_path, fraction):
	midi = pretty_midi.PrettyMIDI(midi_path)
	midi = normalize_instrument_names(midi)
	bpm_st, bpm_value = midi.get_tempo_changes()  
	#time_signatures = midi.time_signature_changes #asumamos 4/4 por ahora

	notes_per_instrument = [x.notes for x in midi.instruments]
	inst_names = [x.name for x in midi.instruments]
	max_time = max([max(x, key=lambda x: x.end) for x in notes_per_instrument], key=lambda x: x.end).end
	bpm_st = np.append(bpm_st, max_time)
	
	measures = get_measures(bpm_value, bpm_st, fraction)
	full_grid = get_time_grid(measures)

	notes_per_instrument = quantize(notes_per_instrument, full_grid)
	instr_measures = group_by_measures(notes_per_instrument, measures)
	remi_values = notes_to_remi(instr_measures, measures)
	#print(remi_values)
	
	re = []
	for i, inst in enumerate(remi_values):
		re.append({"inst_name":inst_names[i], "notes":remi_values[i]})

	return re

if __name__ == "__main__":
	d = EncodedMidiRemi(
		path = "/home/maraneda/music_inpainting_benchmark/data/raw/jsb_chorales/bwv122.6.mid",
		min_length = 16,
		fraction = 24,
		dataset_name = "jsb_chorales"
	).get_tensor()
	#print(len(d[0]["notes"]))
	breakpoint()