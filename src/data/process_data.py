import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = os.path.join(PROJECT_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
INTERIM_DIR = os.path.join(DATA_DIR, "interim")


class Note:
    def __init__(self, row, ticks_per_note):
        self.track = int(row["track"])
        self.measure = int(row["measure"])
        self.rel_pos = round(row["relative_position"] * ticks_per_note)
        self.duration = round(row["duration"] * ticks_per_note)
        self.pitch = int(row["pitch"])
        self.velocity = int(row["velocity"])
        self.tempo = int(row["tempo"])

    def to_noteseq(self):
        return np.array([self.pitch] + [128] * (self.duration - 1))

    def to_remi(self):
        return np.array(
            [
                self.measure,
                self.rel_pos,
                self.duration,
                self.pitch,
                self.velocity,
                self.tempo,
            ]
        )


def to_noteseq(df, ticks_per_note):
    overflow = []
    # groups = df.groupby(by=["track", "measure"]).groups.items()
    last_measures = df.groupby("track").max()["measure"].tolist()
    tracks_ids = list(df.groupby("track").groups.keys())
    tracks = [[] for x in tracks_ids]
    groups = [range(last_measures[track] + 1) for track in tracks_ids]
    for track_ix, measures in enumerate(groups):
        track_df = df[df["track"] == track_ix]
        for m in measures:
            seq = [129 for _ in range(ticks_per_note * 4)]
            notes_df = track_df[track_df["measure"] == m]
            if len(notes_df) <= 0:
                i = 0
                while len(overflow) > 0:
                    if i >= ticks_per_note * 4:
                        break
                    else:
                        seq[i] = overflow.pop(0)
                        i += 1
            else:
                for note in notes_df.iterrows():
                    n = Note(note[1], ticks_per_note)
                    noteseq = n.to_noteseq()
                    # if noteseq se paso pal compas del lado
                    i = 0
                    while len(overflow) > 0:
                        if i >= ticks_per_note * 4:
                            break
                        else:
                            seq[i] = overflow.pop(0)
                            i += 1
                    for i, ix in enumerate(range(n.rel_pos, n.rel_pos + n.duration)):
                        if ix >= ticks_per_note * 4:
                            overflow.append(noteseq[i])
                        else:
                            seq[ix] = noteseq[i]

            tracks[track_ix].append(np.array(seq))
    return np.array(tracks, dtype="object")


def to_remi(df, ticks_per_note):
    groups = df.groupby(by=["track", "measure"]).groups.items()
    tracks = [[] for x in range(list(groups)[-1][0][0] + 1)]
    for keys, indexs in groups:
        seq = []
        for note in df.loc[
            indexs,
        ].iterrows():
            n = Note(note[1], ticks_per_note)
            remi = n.to_remi()
            seq.append(remi)
        tracks[n.track].append(np.array(seq))
    return np.array(tracks, dtype="object")


def make_split(interim_files, dataset, n_context):
    files = os.listdir(interim_files)
    # Dos csvs: measures y contexts
    # filename, track, measure
    # filename, track, measure_st, measure_end

    measures = []
    contexts = []
    logging.info("Generating data splits...")

    for file in tqdm(files):
        npy_file = f"{file.split('.')[0]}.npy"
        path = os.path.join(interim_files, file)
        df = pd.read_csv(path)
        tracks = df["track"].unique()
        max_measures = [df[df["track"] == track]["measure"].max() for track in tracks]

        for track in tracks:
            for measure in range(max_measures[track]):

                measure_entry = {
                    "filename": npy_file,
                    "track": track,
                    "measure": measure,
                }
                measures.append(measure_entry)
                if measure + n_context <= max_measures[track]:
                    context_entry = {
                        "filename": npy_file,
                        "track": track,
                        "measure_st": measure,
                        "measure_end": measure + n_context,
                    }
                    contexts.append(context_entry)

    out_path_measures = os.path.join(DATA_DIR, "splits", f"{dataset}_measures.csv")
    out_path_contexts = os.path.join(DATA_DIR, "splits", f"{dataset}_contexts.csv")

    df_measures = pd.DataFrame(data=measures)
    df_contexts = pd.DataFrame(data=contexts)
    df_measures = add_train_val_test(df_measures)
    df_contexts = add_train_val_test(df_contexts)
    df_measures.to_csv(out_path_measures, index=False)
    df_contexts.to_csv(out_path_contexts, index=False)


def add_train_val_test(df):
    files = df["filename"].unique().tolist()
    random.Random(42).shuffle(files)
    n = len(files)
    train = [{"filename": f, "set": "train"} for f in files[: int(n * 0.8)]]
    val = [
        {"filename": f, "set": "val"}
        for f in files[len(train) : len(train) + int(n * 0.1)]
    ]
    test = [{"filename": f, "set": "test"} for f in files[len(train) + len(val) :]]

    train.extend(val)
    train.extend(test)

    files_sets = pd.DataFrame(data=train)
    merge = pd.merge(df, files_sets, on="filename")
    return merge


def main():
    logging.info("Creating necessary folders...")
    data_sources_file = open(os.path.join(RAW_DIR, "data_sources.json"), "r")
    sources = json.load(data_sources_file)
    for dataset, config in sources.items():
        interim_files = os.path.join(INTERIM_DIR, dataset)
        dataset_dir = os.path.join(PROCESSED_DIR, dataset)
        if not os.path.exists(dataset_dir):
            os.mkdir(dataset_dir)
        n_context = 16
        make_split(interim_files, dataset, n_context)
        for file in tqdm(os.listdir(interim_files)):
            path = os.path.join(interim_files, file)
            df = pd.read_csv(path)
            noteseq_16 = to_noteseq(df, 4)
            noteseq_24 = to_noteseq(df, 6)
            remi_16 = to_remi(df, 4)
            remi_24 = to_remi(df, 6)
            data = [noteseq_16, noteseq_24, remi_16, remi_24]
            formats = ["noteseq_16", "noteseq_24", "remi_16", "remi_24"]
            for f, d in zip(formats, data):
                out_dir = os.path.join(dataset_dir, f)
                if not os.path.exists(out_dir):
                    os.mkdir(out_dir)
                out_path = os.path.join(out_dir, f"{file.split('.')[0]}.npy")
                np.save(out_path, d, allow_pickle=True, fix_imports=False)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
