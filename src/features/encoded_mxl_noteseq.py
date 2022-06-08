import argparse
import torch
import os 
import pretty_midi
import random
import numpy as np
import music21
from pathlib import Path

class EncodedMxlNoteSeq:
    def __init__(self, path, fraction, dataset_name, min_length):
        self.path = path
        self.min_length = min_length
        self.fraction = fraction
        self.dataset_name = dataset_name

    def get_tensor(self):
        data = music21.converter.parse(self.path)

        metadatas = [
            FermataMetadata(),
            TickMetadata(subdivision=4),
            KeyMetadata()
        ]

        score_tensor, metadata_tensor = make_tensors(data, metadatas)
        return score_tensor, metadata_tensor


def make_tensors(chorale, metadatas):
        
    # todo check on chorale with Chord
    subdivision=4

    chorale_tensor = [part_to_tensor(part, i, 0.0, part.flat.highestTime) for i, part in enumerate(chorale.parts)]
    chorale_tensor = torch.cat(chorale_tensor, 0)

    metadata_tensor = get_metadata_tensor(chorale, metadatas)
    
    #offsetStart = 0.0
    #offsetEnd = chorale.flat.highestTime
    #start_tick = int(offsetStart * subdivision)
    #end_tick = int(offsetEnd * subdivision)

    #local_chorale_tensor = self.extract_score_tensor_with_padding(chorale_tensor, start_tick, end_tick)
    #local_metadata_tensor = self.extract_metadata_with_padding(metadata_tensor, start_tick, end_tick)

    return chorale_tensor, metadata_tensor
    


def part_to_tensor(part, part_id, offsetStart, offsetEnd):
        """
        :param part:
        :param part_id:
        :param offsetStart:
        :param offsetEnd:
        :return: torch IntTensor (1, length)
        """
        list_notes_and_rests = [x if type(x)==music21.note.Note else x for x in part.flat if type(x)==music21.note.Note or type(x)==music21.note.Rest] #reemplazo no dependiente del offset de inicio y fin
        subdivision = 4
        length = int((offsetEnd - offsetStart) * subdivision)  # in ticks
        min_length = 16
        length = max(length, min_length*16)

        # construct sequence
        j = 0
        i = 0
        t = np.zeros((length, 2))
        is_articulated = True
        num_notes = len(list_notes_and_rests)
        while i < length:
            if j < num_notes - 1:
                if (list_notes_and_rests[j + 1].offset > i / subdivision + offsetStart):
                    t[i, :] = [list_notes_and_rests[j].pitch.midi if type(list_notes_and_rests[j])==music21.note.Note else 129, is_articulated]
                    i += 1
                    is_articulated = False
                else:
                    j += 1
                    is_articulated = True
            else:
                t[i, :] = [list_notes_and_rests[j].pitch.midi if type(list_notes_and_rests[j])==music21.note.Note else 129, is_articulated]
                i += 1
                is_articulated = False
        SLUR_SYMBOL = 128
        seq = t[:, 0] * t[:, 1] + (1 - t[:, 1]) * SLUR_SYMBOL
        tensor = torch.from_numpy(seq).long()[None, :]
        return tensor


def get_metadata_tensor(score, metadatas):
        """
        Adds also the index of the voices
        :param score: music21 stream
        :return:tensor (num_voices, chorale_length, len(self.metadatas) + 1)
        """
        subdivision=4
        num_voices=4
        md = []
        for metadata in metadatas:
            sequence_metadata = torch.from_numpy(metadata.evaluate(score, subdivision)).long().clone()
            square_metadata = sequence_metadata.repeat(num_voices, 1)
            md.append(square_metadata[:, :, None])
        chorale_length = int(score.duration.quarterLength * subdivision)

        # add voice indexes
        voice_id_metada = torch.from_numpy(np.arange(num_voices)).long().clone()
        square_metadata = torch.transpose(voice_id_metada.repeat(chorale_length, 1), 0, 1)
        md.append(square_metadata[:, :, None])

        all_metadata = torch.cat(md, 2)
        return all_metadata


def extract_score_tensor_with_padding(tensor_score, start_tick, end_tick):
        """
        :param tensor_chorale: (num_voices, length in ticks)
        :param start_tick:
        :param end_tick:
        :return: tensor_chorale[:, start_tick: end_tick]
        with padding if necessary
        i.e. if start_tick < 0 or end_tick > tensor_chorale length
        """
        assert start_tick < end_tick
        assert end_tick > 0
        length = tensor_score.size()[1]

        padded_chorale = []
        # todo add PAD_SYMBOL
        if start_tick < 0:
            start_symbols = np.array([note2index[START_SYMBOL]
                                      for note2index in self.note2index_dicts])
            start_symbols = torch.from_numpy(start_symbols).long().clone()
            start_symbols = start_symbols.repeat(-start_tick, 1).transpose(0, 1)
            #print("start",start_symbols)
            padded_chorale.append(start_symbols)

        slice_start = start_tick if start_tick > 0 else 0
        slice_end = end_tick if end_tick < length else length

        padded_chorale.append(tensor_score[:, slice_start: slice_end])
        #print("score info", tensor_score, tensor_score.shape, slice_start, slice_end)
        
        if end_tick > length:
            end_symbols = np.array([note2index[END_SYMBOL]
                                    for note2index in self.note2index_dicts])
            end_symbols = torch.from_numpy(end_symbols).long().clone()
            end_symbols = end_symbols.repeat(end_tick - length, 1).transpose(0, 1)
            padded_chorale.append(end_symbols)

        padded_chorale = torch.cat(padded_chorale, 1)
        #print("padded", padded_chorale)
        return padded_chorale



class FermataMetadata():
    """
    Metadata class which indicates if a fermata is on the current note
    """

    def __init__(self):
        super(FermataMetadata, self).__init__()
        self.is_global = False
        self.num_values = 2
        self.name = 'fermata'

    def evaluate(self, chorale, subdivision):
        part = chorale.parts[0]
        
        length = int(part.duration.quarterLength * subdivision)  # in 16th notes
        list_notes = part.flat.notes
        num_notes = len(list_notes)
        j = 0
        i = 0
        fermatas = np.zeros((length,))
        # Se acomoda la lista de notas (y la propiedad de fermata) a la cuantizacion del tiempo
        while i < length:
            if j < num_notes - 1:
                if list_notes[j + 1].offset > i / subdivision:
                    if len(list_notes[j].expressions) == 1: #Esta linea define las fermatas que vienen de la libreria
                        fermata = True
                    else:
                        fermata = False
                    fermatas[i] = fermata
                    i += 1
                else:
                    j += 1
            else:
                if len(list_notes[j].expressions) == 1:
                    fermata = True
                else:
                    fermata = False

                fermatas[i] = fermata
                i += 1

        return np.array(fermatas, dtype=np.int32)



class KeyMetadata():
    """
    Metadata class that indicates in which key we are
    Only returns the number of sharps or flats
    Does not distinguish a key from its relative key
    """

    def __init__(self, window_size=4):
        super(KeyMetadata, self).__init__()
        self.window_size = window_size
        self.is_global = False
        self.num_max_sharps = 7
        self.num_values = 16
        self.name = 'key'

    def get_index(self, value):
        """
        :param value: number of sharps (between -7 and +7)
        :return: index in the representation
        """
        return value + self.num_max_sharps + 1

    def get_value(self, index):
        """
        :param index:  index (between 0 and self.num_values); 0 is unused (no constraint)
        :return: true number of sharps (between -7 and 7)
        """
        return index - 1 - self.num_max_sharps

    def evaluate(self, chorale, subdivision):
        # init key analyzer
        # we must add measures by hand for the case when we are parsing midi files
        chorale_with_measures = music21.stream.Score()
        for part in chorale.parts:
            chorale_with_measures.append(part.makeMeasures())

        ka = music21.analysis.floatingKey.KeyAnalyzer(chorale_with_measures)
        ka.windowSize = self.window_size
        res = ka.run()

        measure_offset_map = chorale_with_measures.parts.measureOffsetMap()
        length = int(chorale.duration.quarterLength * subdivision)  # in 16th notes

        key_signatures = np.zeros((length,))

        measure_index = -1
        for time_index in range(length):
            beat_index = time_index / subdivision
            if beat_index in measure_offset_map:
                measure_index += 1
                if measure_index == len(res):
                    measure_index -= 1

            key_signatures[time_index] = self.get_index(res[measure_index].sharps)
            #print(np.array(key_signatures, dtype=np.int32))
        return np.array(key_signatures, dtype=np.int32)


class TickMetadata():
    """
    Metadata class that tracks on which subdivision of the beat we are on
    """

    def __init__(self, subdivision):
        super(TickMetadata, self).__init__()
        self.is_global = False
        self.num_values = subdivision
        self.name = 'tick'

    def get_index(self, value):
        return value

    def get_value(self, index):
        return index

    def evaluate(self, chorale, subdivision):
        assert subdivision == self.num_values
        # suppose all pieces start on a beat
        length = int(chorale.duration.quarterLength * subdivision)
        return np.array(list(map(
            lambda x: x % self.num_values,
            range(length)
        )))

if __name__ == "__main__":
	d = EncodedMxlNoteSeq(
		path = "/home/maraneda/music_inpainting_benchmark/data/external/jsb_chorales/bwv122.6.mxl",
		min_length = 16,
		fraction = 16,
		dataset_name = "jsb_chorales"
	).get_tensor()
	breakpoint()
