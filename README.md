# Music Inpainting Benchmark

Benchmark of existing inpainting solutions:


## Models

| Model                     | Year | Repo                                                                |                                Paper                                |
|---------------------------|:----:|---------------------------------------------------------------------|:-------------------------------------------------------------------:|
| DeepBach                  | 2017 | <a href="https://github.com/Ghadjeres/DeepBach">Repo</a>                               | <a href="https://arxiv.org/pdf/1612.01010.pdf">Paper</a>                                |
| CocoNet                   | 2017 | <a href="https://github.com/magenta/magenta/tree/main/magenta/models/coconet">Repo</a> | <a href="https://arxiv.org/pdf/1903.07227.pdf">Paper</a>                                |
| AnticipationRNN           | 2018 | <a href="https://github.com/Ghadjeres/Anticipation-RNN">Repo</a>                       | <a href="https://link.springer.com/content/pdf/10.1007/s00521-018-3868-4.pdf">Paper</a> |
| InpaintNet                | 2019 | <a href="https://github.com/ashispati/InpaintNet">Repo</a>                             | <a href="https://archives.ismir.net/ismir2019/paper/000040.pdf">Paper</a>               |
| Music SketchNet           | 2020 | <a href="https://github.com/RetroCirce/Music-SketchNet">Repo</a>                       | <a href="https://arxiv.org/pdf/2008.01291.pdf">Paper</a>                                |
| Variable Length Infilling | 2021 | <a href="https://github.com/reichang182/variable-length-piano-infilling">Repo</a>      | <a href="https://arxiv.org/pdf/2108.05064.pdf">Paper</a>                                |


## Datasets

| Dataset      | Size  | Description             |                                         Source                                        | Paper                                |             Type            |
|--------------|-------|-------------------------|:-------------------------------------------------------------------------------------:|--------------------------------------|:---------------------------:|
| AILabs       | 1747  | Live Piano Performances |  <a href="https://github.com/YatingMusic/compound-word-transformer/blob/main/dataset/Dataset.md">Source</a>  | <a href="https://arxiv.org/pdf/2101.02402.pdf">Paper</a> | Single Instrument Polyphony |
| JSB Chorales | 385   | Bach Chorales Scores    | <a href="https://github.com/cuthbertLab/music21/tree/master/music21/corpus/bach">Source</a>                | -                                    | Fixed Voices Polyphony      |
| IrishFolk    | 45849 | Irish Folk Songs        | <a href="https://github.com/IraKorshunova/folk-rnn/tree/master/data">Source</a>                            | <a href="https://arxiv.org/pdf/1604.08723.pdf">Paper</a> | Monophony                   |

## Data Representation


### Music SketchNet

```
DEFAULT_FRACTION: 24

# 0-127 note, 128 hold, 129 rest
note_seq: [
    [48, 128, 128, 128, 128, 128, 50, 128, 128, 128, 128, 128, 52, 128, 128, 128, 128, 128, 53, 128, 128, 128, 128, 128]
]

# [px, rx, len_x, nrx, gd]
factorized: [
    [48, 50, 52, 53, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128],
    [1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2],
    [4],
    [[1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]],
    [48, 128, 128, 128, 128, 128, 50, 128, 128, 128, 128, 128, 52, 128, 128, 128, 128, 128, 53, 128, 128, 128, 128, 128]
]

model_input: [n_batch, **REVISAR**]
model_output: [n_batch, n_measures_middle, DEFAULT_FRACTION, n_classes]

```


### DeepBach

```
index2note:
    {0: 'D#5', 1: 'E-4', 2: 'E-5', 3: 'rest', 4: 'F#4', 5: 'E#5', 6: 'G#4', 7: 'B4', 8: 'D4', 9: 'A5', 10: 'END', 11: 'G-4', 12: 'C#5', 13: 'G4', 14: 'A3', 15: 'D#4', 16: 'START', 17: 'D5', 18: 'C5', 19: 'F5', 20: 'A-4', 21: 'C4', 22: 'C#4', 23: 'E5', 24: 'E#4', 25: 'A#4', 26: 'D-5', 27: 'E4', 28: 'G-5', 29: 'A-5', 30: 'A4', 31: 'G5', 32: 'B-4', 33: 'F#5', 34: '__', 35: 'F4', 36: 'OOR', 37: 'G#5', 38: 'B3'}

score_tensor = tensor([[36, 34, 34, 34, 36, 34, 34, 34, 34, 34, 34, 34,  8, 34, 34, 34, 38, 34,
         34, 34, 34, 34, 14, 34, 36, 34, 34, 34, 36, 34, 34, 34, 34, 34, 14, 34,
         38, 34, 34, 34, 14, 34, 34, 34, 34, 34, 34, 34, 38, 34, 34, 34,  8, 34,
         34, 34, 34, 34, 34, 34, 21, 34, 34, 34, 38, 34, 34, 34, 14, 34, 34, 34,
         34, 34, 34, 34, 36, 34, 34, 34, 34, 34, 34, 34, 38, 34, 34, 34, 38, 34,
         34, 34, 21, 34, 34, 34,  8, 34, 34, 34,  8, 34, 34, 34, 34, 34, 21, 34,
         38, 34, 34, 34, 14, 34, 34, 34, 34, 34, 34, 34, 36, 34, 34, 34, 38, 34,
         34, 34, 34, 34, 34, 34, 21, 34, 34, 34,  8, 34, 34, 34, 34, 34, 34, 34,
         21, 34, 34, 34, 38, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 36, 34,
         34, 34, 34, 34, 34, 34, 38, 34, 34, 34,  8, 34, 34, 34, 34, 34, 34, 34,
         21, 34, 34, 34, 38, 34, 34, 34, 34, 34, 34, 34, 14, 34, 34, 34, 36, 34,
         34, 34, 34, 34, 14, 34, 38, 34, 34, 34, 14, 34, 34, 34, 34, 34, 34, 34,
         38, 34, 34, 34,  8, 34, 34, 34, 34, 34, 34, 34, 21, 34, 34, 34, 38, 34,
         34, 34, 14, 34, 34, 34, 34, 34, 34, 34, 36, 34, 34, 34, 34, 34, 34, 34]])

# Metadata = [Fermata, Tick, Key, N_Voice]
metadata_tensor = tensor([[ 0,  0, 15,  0],
        [ 0,  1, 15,  0],
        [ 0,  2, 15,  0],
        [ 0,  3, 15,  0],
        [ 0,  0, 15,  0],
        [ 0,  1, 15,  0],
        [ 0,  3, 15,  0],
        [ 0,  0, 15,  0],
        [ 1,  2, 15,  0],
        [ 1,  3, 15,  0],
        [ 1,  0, 15,  0],
        [ 1,  1, 15,  0],
        [ 1,  2, 15,  0],
        [ 1,  3, 15,  0]])
```
------------

Project Organization
------------

    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── interim        <- CSV data containing all notes coming from raw sources. Intermediate before vectorization.
    │   ├── processed      <- Vectorization of data ready to feed the models.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained models weights.
    │
    ├── results            <- Generated results, tables, csvs, etc.
    │   └── images         <- Generated graphics and figures
    │
    ├── environment.yaml   <- Libraries and modules required by the environment to reproduce the project.
    │
    └── src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        │
        ├── data           <- Scripts to download or generate data
        │   ├── download_from_souce.py
        │   ├── standardize_data.py
        │   └── process_data.py
        │
        ├── features       <- Scripts to turn raw data into features for modeling
        │   └── build_features.py
        │
        ├── models         <- Scripts to train models and generate new data.
        │   ├── sketchnet.py
        │   ├── inpaintnet.py
        │   ├── arnn.py
        │   └── vli.py
        │
        ├── metrics         <- Scripts to calculation of metrics.
        │
        └── visualization  <- Scripts to create exploratory and results oriented visualizations
            └── plot_metrics.py


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>.</small></p>
