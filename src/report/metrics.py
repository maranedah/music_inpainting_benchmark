import math
from turtle import position
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import torch

loss = nn.NLLLoss()

class Metrics:
    def __init__(self):
        
        self.accuracy = 0
        self.precision = 0
        self.recall = 0
        self.f1 = 0
        self.perplexity = 0
        self.loss = 0
        

        self.pos_f1 = 0
        self.px_acc = 0
        self.rx_acc = 0
        self.true_silence = []
        self.pred_silence = []
        self.silence_divergence = None
        self.true_px_entropy_similarity = []
        self.pred_px_entropy_similarity = []
        self.px_similarity_div = None
        self.true_groove_similarity = []
        self.pred_groove_similarity = []
        self.groove_similarity_div = None

        self.i = 0

    def get_normalized_hist(self, data, resolution=1000):
        return torch.from_numpy(np.histogram(data, bins=[i/resolution for i in range(resolution)])[0]/self.i)

    def pred_silence_hist(self): 
        return torch.from_numpy(np.histogram(self.pred_silence, bins=[i/1000 for i in range(1000)])[0]/self.i)

    def calc_metrics(self, y_true, y_pred, y_past = None, y_future = None):
        if len(y_pred) == 0:
            return None
        self.i += 1
        if len(y_true.shape) == 2:
            # Remove masked stuff
            y_true = y_true[np.sum(y_true, axis=1) > 0]
            y_pred = y_pred[np.sum(y_pred, axis=1) > 0]
            remi_true = y_true
            remi_pred = y_pred
            if y_past is not None and y_future is not None:
                noteseq_true = remi2noteseq(y_true, start_bar=6, bar_length=4, measure_size=16)
                noteseq_pred = remi2noteseq(y_pred, start_bar=6, bar_length=4, measure_size=16)
                noteseq_past = remi2noteseq(y_past, start_bar=0, bar_length=6, measure_size=16)
                noteseq_future = remi2noteseq(y_future, start_bar=10, bar_length=6, measure_size=16)
                remi_past = y_past
                remi_future = y_future
                measure_size=16
                
            
        else: #if not shape len 2
            noteseq_true = y_true
            noteseq_pred = y_pred
            if y_true.shape[0] == 24:
                remi_true = noteseq2remi(y_true, time_steps=24, start_bar=0, bar_length=1)
                remi_pred = noteseq2remi(y_pred, time_steps=24, start_bar=0, bar_length=1)
            else:
                remi_true = noteseq2remi(y_true, time_steps=24, start_bar=6, bar_length=4)
                remi_pred = noteseq2remi(y_pred, time_steps=24, start_bar=6, bar_length=4)
            if y_past is not None and y_future is not None:
                noteseq_past = y_past
                noteseq_future = y_future
                remi_past = noteseq2remi(y_past, time_steps=24, start_bar=0, bar_length=6) 
                remi_future = noteseq2remi(y_future, time_steps=24, start_bar=10, bar_length=6)
                measure_size=24

        pos_f1 = position_score(remi_true, remi_pred)
        px_acc = poly_px_acc(remi_true, remi_pred)
        rx_acc = poly_rx_acc(remi_true, remi_pred)

        self.pos_f1 += pos_f1
        self.rx_acc += rx_acc
        self.px_acc += px_acc
        
        if y_past is not None and y_future is not None:
            self.true_silence.append(silence_percent(noteseq_true))
            self.pred_silence.append(silence_percent(noteseq_pred))
            self.silence_divergence = self.js_div(self.true_silence, self.pred_silence)

            self.true_px_entropy_similarity.append(mean_px_entropy_similarity(remi_true, remi_past, remi_future))
            self.pred_px_entropy_similarity.append(mean_px_entropy_similarity(remi_pred, remi_past, remi_future))
            self.px_similarity_div = self.js_div(self.true_px_entropy_similarity, self.pred_px_entropy_similarity)

            self.true_groove_similarity.append(groove_similarity(remi_true, remi_past, remi_future, measure_size=measure_size))
            self.pred_groove_similarity.append(groove_similarity(remi_pred, remi_past, remi_future, measure_size=measure_size))
            self.groove_similarity_div = self.js_div(self.true_groove_similarity, self.pred_groove_similarity)

        return self.pos_f1, self.px_acc, self.rx_acc, self.silence_divergence, self.px_similarity_div, self.groove_similarity_div
        

    def reset_metrics(self):
        self.pos_f1 = 0
        self.px_acc = 0
        self.rx_acc = 0
        self.true_silence = []
        self.pred_silence = []
        self.silence_divergence = None
        self.true_px_entropy_similarity = []
        self.pred_px_entropy_similarity = []
        self.px_similarity_div = None
        self.true_groove_similarity = []
        self.pred_groove_similarity = []
        self.groove_similarity_div = None
        self.i = 0

    def js_div(self, true_attrs, pred_attrs):
        Q = self.get_normalized_hist(true_attrs, 100)
        P = self.get_normalized_hist(pred_attrs, 100)
        M = 0.5 * (Q+P) 
        js_div = 0.5* (torch.nan_to_num(Q * (Q / M).log()).sum() + torch.nan_to_num(P * (P / M).log()).sum())
        return js_div.item()

    def pitch_similarity(self, y_true, y_pred):
        num_pitch = np.sum(y_true < 128)
        acc = np.sum(np.logical_and(y_true == y_pred,  y_true < 128)) / num_pitch
        return acc


    def rhythm_similarity(self, y_true, y_pred):
        num_pitch = np.sum(y_true >= 128)
        acc = np.sum(np.logical_and(y_true == y_pred,  y_true >= 128)) / num_pitch
        return acc
        

    

    def __str__(self):
        return f"{self.get_px_acc()},{self.get_rx_acc()},{self.get_precision()},{self.get_recall()},{self.get_f1()}\n"

    def get_precision(self):
        return self.precision/max(1, self.i)

    def get_recall(self):
        return self.recall/max(1, self.i)

    def get_accuracy(self):
        return self.accuracy/max(1, self.i)

    def get_f1(self):
        return self.f1/max(1, self.i)

    def get_px_acc(self):
        return self.px_acc/max(1, self.i)

    def get_rx_acc(self):
        return self.rx_acc/max(1, self.i)

    def get_loss(self):
        return self.loss/max(1, self.i)

    def get_perplexity(self):
        return math.exp(self.loss/max(1, self.i))

    def get_pos_f1(self):
        return self.pos_f1/max(1, self.i)

############# metricas f1 #############




def position_score(y_true, y_pred):
    if len(y_pred)==0: return 0
    y_true_pos = np.array(y_true)[:,1:3]
    y_pred_pos = np.array(y_pred)[:,1:3]
    y_true_pos = [tuple(elem) for elem in y_true_pos]
    y_pred_pos = [tuple(elem) for elem in y_pred_pos]
    position_f1 = f1(y_true_pos, y_pred_pos)
    return position_f1

def pitch_accuracy(y_true, y_pred):
    if len(y_pred)==0: return 0
    y_true_pos = np.array(y_true)[:,1:3]
    y_pred_pos = np.array(y_pred)[:,1:3]
    y_true_pos = [tuple(elem) for elem in y_true_pos]
    y_pred_pos = [tuple(elem) for elem in y_pred_pos]
    _, shared_positions = true_positives(y_true_pos, y_pred_pos)
    y_true_pitch = [elem[3] for elem in y_true if tuple([elem[1], elem[2]]) in shared_positions]
    y_pred_pitch = [elem[3] for elem in y_pred if tuple([elem[1], elem[2]]) in shared_positions]
    if len(y_true_pitch) == 0:
        return 0
    return sum([ int(true_pitch == pred_pitch) for true_pitch, pred_pitch in zip(y_true_pitch, y_pred_pitch)])/len(y_true_pitch)

def poly_px_acc(y_true, y_pred):
    if len(y_pred)==0: return 0
    y_true = sorted(y_true, key=lambda x: tuple([x[1],x[2]])) #importante por alguna razon
    y_pred = sorted(y_pred, key=lambda x: tuple([x[1],x[2]])) #importante por alguna razon
    y_true_pos = np.array(y_true)[:,1:3]
    y_pred_pos = np.array(y_pred)[:,1:3]
    y_true_pos = [tuple(elem) for elem in y_true_pos]
    y_pred_pos = [tuple(elem) for elem in y_pred_pos]

    _, shared_positions = true_positives(y_true_pos, y_pred_pos)
    shared_positions = sorted(list(set(shared_positions)))
    y_true = [x for x in y_true if tuple([x[1],x[2]]) in shared_positions]
    y_pred = [x for x in y_pred if tuple([x[1],x[2]]) in shared_positions]
    true_pitches_per_onset = [[a[3] for a in list(group)] for _, group in groupby(y_true, lambda x: tuple([x[1], x[2]]))]
    pred_pitches_per_onset = [[a[3] for a in list(group)] for _, group in groupby(y_pred, lambda x: tuple([x[1], x[2]]))]
    aciertos = [[pitch in pred_pitches_per_onset[i] for pitch in elem] for i, elem in enumerate(true_pitches_per_onset)]
    flatten_aciertos = [int(item) for sublist in aciertos for item in sublist]
    return sum(flatten_aciertos)/max(1,len(flatten_aciertos))

def poly_rx_acc(y_true, y_pred):
    if len(y_pred)==0: return 0
    y_true = sorted(y_true, key=lambda x: tuple([x[1],x[2]])) #importante por alguna razon
    y_pred = sorted(y_pred, key=lambda x: tuple([x[1],x[2]])) #importante por alguna razon
    y_true_pos = np.array(y_true)[:,1:3]
    y_pred_pos = np.array(y_pred)[:,1:3]
    y_true_pos = [tuple(elem) for elem in y_true_pos]
    y_pred_pos = [tuple(elem) for elem in y_pred_pos]

    _, shared_positions = true_positives(y_true_pos, y_pred_pos)
    shared_positions = sorted(list(set(shared_positions)))
    y_true = [x for x in y_true if tuple([x[1],x[2]]) in shared_positions]
    y_pred = [x for x in y_pred if tuple([x[1],x[2]]) in shared_positions]
    true_rhythm_per_onset = [[a[4] for a in list(group)] for _, group in groupby(y_true, lambda x: tuple([x[1], x[2]]))]
    pred_rhythm_per_onset = [[a[4] for a in list(group)] for _, group in groupby(y_pred, lambda x: tuple([x[1], x[2]]))]
    aciertos = [[rhythm in pred_rhythm_per_onset[i] for rhythm in elem] for i, elem in enumerate(true_rhythm_per_onset)]
    flatten_aciertos = [int(item) for sublist in aciertos for item in sublist]
    return sum(flatten_aciertos)/max(1,len(flatten_aciertos))

def rhythm_accuracy(y_true, y_pred):
    if len(y_pred)==0: return 0
    y_true_pos = np.array(y_true)[:,1:3]
    y_pred_pos = np.array(y_pred)[:,1:3]
    y_true_pos = [tuple(elem) for elem in y_true_pos]
    y_pred_pos = [tuple(elem) for elem in y_pred_pos]
    _, shared_positions = true_positives(y_true_pos, y_pred_pos)
    y_true_rhythm = [elem[4] for elem in y_true if tuple([elem[1], elem[2]]) in shared_positions]
    y_pred_rhythm = [elem[4] for elem in y_pred if tuple([elem[1], elem[2]]) in shared_positions]
    if len(y_true_rhythm) == 0:
        return 0
    return sum([ int(true_pitch == pred_pitch) for true_pitch, pred_pitch in zip(y_true_rhythm, y_pred_rhythm)])/len(y_true_rhythm)

def true_positives(y_true, y_pred):
    true_positives = [elem for elem in y_true if elem in y_pred]
    return len(true_positives), true_positives
def false_negatives(y_true, y_pred):
    false_negatives = [elem for elem in y_true if elem not in y_pred]
    return len(false_negatives)

def false_positives(y_true, y_pred):
    c1 = Counter(y_true)
    c2 = Counter(y_pred)
    diff = c2-c1
    return sum(diff.values())

def precision(y_true, y_pred):
    tp, _ = true_positives(y_true, y_pred)
    fp = false_positives(y_true, y_pred)
    if tp+fp == 0:
        return 0
    return tp / (tp+fp)

def recall(y_true, y_pred):
    tp, _ = true_positives(y_true, y_pred)
    fn = false_negatives(y_true, y_pred)
    if tp+fn == 0:
        return 0
    return tp / (tp+fn)

def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    if p+r == 0:
        return 0
    return 2 * (p*r)/(p+r)


###### metricas ###############

from collections import Counter

def silence_percent(noteseq):
  d = Counter(noteseq)
  return d[129]/len(noteseq)


# From Jazz Transformer metrics:
# - https://arxiv.org/abs/2008.01307
# If a piece’s tonality is clear, a fixed group of pitch classes should  dominate the  pitch histogram  resulting in a low entropy
def pitch_entropy(measure):
    measure = np.array(measure)
    pitches = measure[:,3]
    pitch_classes = pitches % 12
    h, _ = np.histogram(pitch_classes, bins=12, range=[0,12])
    h = h / len(measure) #normalize over n_notes
    h = h[np.nonzero(h)]
    r = -sum(h * np.log(h))
    return r

def pitch_entropy_similarity(measure1, measure2):
  return np.abs(pitch_entropy(measure1) - pitch_entropy(measure2))

from itertools import groupby
def group_measures(data):
  return [list(group) for key, group in groupby(data, lambda x: x[1])]

def mean_px_entropy_similarity(data, y_past, y_future):
    c_m = group_measures(data)
    c_p = group_measures(y_past)
    c_f = group_measures(y_future)

    res = 0
    for m1 in c_m:
        for m2 in c_p:
            res += pitch_entropy_similarity(m1, m2)

    for m1 in c_m:
        for m2 in c_f:
            res += pitch_entropy_similarity(m1, m2)
    return res/max(1,(len(c_m)*(len(c_p) + len(c_f))))

# From Jazz Transformer metrics:
# - https://arxiv.org/abs/2008.01307
# The positions in a bar at which there is at least a note onset. Requires time-step representation
def grooving_similarity(seq1, seq2):
  assert len(seq1) == len(seq2)
  Q = len(seq1)
  return 1 - (1/Q) * sum(np.logical_xor(seq1, seq2))

def groove_similarity(data, y_past, y_future, measure_size):
    c_m = group_measures(data)
    c_p = group_measures(y_past)
    c_f = group_measures(y_future)

    res = 0
    for m1 in c_m:
        for m2 in c_p:
            m1_ = remi2noteseq(m1, start_bar=m1[0][1], bar_length=1, measure_size=measure_size)
            m2_ = remi2noteseq(m2, start_bar=m2[0][1], bar_length=1, measure_size=measure_size)
            res += grooving_similarity(get_onsets(m1_), get_onsets(m2_))

    for m1 in c_m:
        for m2 in c_f:
            m1_ = remi2noteseq(m1, start_bar=m1[0][1], bar_length=1, measure_size=measure_size)
            m2_ = remi2noteseq(m2, start_bar=m2[0][1], bar_length=1, measure_size=measure_size)
            res += grooving_similarity(get_onsets(m1_), get_onsets(m2_))
    return res/max(1,(len(c_m)*(len(c_p) + len(c_f))))




######### transformaciones ##############

def get_onsets(noteseq):
    return torch.where(torch.tensor(noteseq)>=128, 0, 1)

#time_steps = 16 or 24
def noteseq2remi(data, time_steps, start_bar, bar_length):
    onsets = torch.where(torch.tensor(data)>=128, 0, 1)
    onsets = torch.nonzero(onsets).flatten().tolist() #onsets indexes
    onsets.append(time_steps*bar_length)
    
    notes = []
    for i in range(len(onsets)-1):
        onset = onsets[i]
        note = [-1, -1, -1, -1, -1, -1]
        note[1] = onset//time_steps + start_bar #measure
        note[2] = onset%time_steps #position
        note[3] = data[onset] #pitch
        
        duration = 0
        j = onset
        while True:
            if j == onsets[i+1]:
                break
            if data[j] == 129:
                break
            if j < onsets[i+1]:
                duration += 1
            j+=1
        
        note[4] = duration - 1 #duration
        notes.append(note)
    return notes

def remi2noteseq(data, start_bar, bar_length, measure_size):
    ctxt = [129 for i in range(measure_size*bar_length)]
    for note in data:
        i_note = (note[1]-start_bar)*measure_size + note[2]
        if i_note>=len(ctxt):
            continue
        ctxt[i_note] = note[3]
        ctxt[i_note+1:i_note+1+note[4]] = [128 for i in range(note[4])]
    ctxt = ctxt[:measure_size*bar_length]
    return ctxt

