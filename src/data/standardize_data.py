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
INTERIM_DIR = os.path.join(DATA_DIR, "interim")


class StandardizedNote():
    def __init__(self, note, track, data, ticks_per_beat):
        self.track = track
        self.time = self.scaled_time(note, ticks_per_beat, data.resolution)
        self.duration = self.scaled_duration(note, ticks_per_beat, data.resolution)
        self.pitch = note.pitch
        self.velocity = note.velocity 
        self.tempo = self.get_tempo(note, data.tempos)
        self.time_signature = self.get_time_signature(note, data.time_signatures)
        self.ticks_per_measure = self.get_ticks_per_measure(ticks_per_beat)
        self.measure = self.get_measure(note, data.barlines)

    def get_measure(self, note, barlines):
        measure = len(barlines) - len([m for m in barlines if note.time<m.time]) -1
        return measure

    def scaled_time(self, note, ticks_per_beat, resolution):
        return round(note.time*ticks_per_beat/resolution)

    def scaled_duration(self, note, ticks_per_beat, resolution):
        return round(note.duration*ticks_per_beat/resolution)

    def get_tempo(self, note, tempos):
        last_tempo = sorted([tempo for tempo in tempos if note.time>=tempo.time], key=lambda x: x.time)[-1]
        return last_tempo.qpm

    def get_time_signature(self, note, time_signatures):    
        last_ts = sorted([ts for ts in time_signatures if note.time>=ts.time], key=lambda x: x.time)[-1]
        return last_ts.numerator, last_ts.denominator

    def get_ticks_per_measure(self, ticks_per_beat):
        return int(self.time_signature[0]*ticks_per_beat*4/self.time_signature[1])




def main():
    logging.info('Creating necessary folders...')
    data_sources_file = open(os.path.join(RAW_DIR, "data_sources.json"), "r")
    sources = json.load(data_sources_file)
    for dataset, config in sources.items():
        raw_files = os.path.join(RAW_DIR, dataset)
        out_dir = os.path.join(INTERIM_DIR, dataset)
        if not os.path.exists(out_dir): os.mkdir(out_dir)
        for file in tqdm(os.listdir(raw_files)):
            path = os.path.join(raw_files, file)
            try:
                if ".mxl" in file:
                    data = muspy.read_musicxml(path)
                if ".mid" in file:
                    data = muspy.read_midi(path)
            except IOError as e:
                continue

            if len(data.tempos) == 0:
                data.tempos = [muspy.Tempo(time=0.0, qpm=120)]
            #breakpoint()

            ticks_per_beat = 24
            data.infer_barlines()
            notes = [
                StandardizedNote(
                    note, 
                    i, 
                    data,
                    ticks_per_beat, 
                )
                for i, track in enumerate(data.tracks)
                for note in track.notes
            ]

            notes = [
                [
                    note.track,
                    note.measure,
                    note.time,
                    note.duration,
                    note.pitch,
                    note.velocity,
                    note.tempo,
                    f"{note.time_signature[0]}/{note.time_signature[1]}",
                    note.ticks_per_measure
                ]
                for note in notes
            ]

            df = pd.DataFrame(
                columns=[
                    #"resolution",
                    "track",
                    "measure",
                    "time",
                    "duration",
                    "pitch",
                    "velocity",
                    "tempo",
                    "time_signature",
                    "ticks_per_measure"
                ],
                data=notes
            )
            out_path = path = os.path.join(out_dir, f"{file.split('.')[0]}.csv")
            df.to_csv(out_path, index=False)


            



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()