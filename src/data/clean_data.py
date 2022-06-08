import pandas as pd
import argparse
import os 
import pretty_midi
import numpy as np
import random
from pathlib import Path



# Usar los DF para filtrar los datos que vamos a usar
# - Non empty
# - Only 4/4
# - Is Monophony 
# - Non Repeated
# - Min Length

def non_empty(df): 
	return df[df["is_empty"]==0]

def non_repeated(df): 
	bool_series = df["hash"].duplicated(keep='first')
	return df[~bool_series]

def only_4_4(df):
	return df[df["is_4_4"]==1]

def only_monophony(df):
	return df[df["is_monophony"]==(True,)]

def min_length(df, min_length):
	return df[df["n_measures"]>=min_length]

def has_four_voices(df):
	return df[df["n_instruments"]==4]

def all_monophony(df):
	pass

def clean_df(df, dataset):
	if dataset == "folk":
		# df = match_existing(another_df)
		df = non_empty(df)
		df = non_repeated(df)
		df = only_4_4(df)
		df = only_monophony(df)
		df = min_length(df, min_length=17)
		
	elif dataset == "jsb_chorales":
		df = non_empty(df)
		df = non_repeated(df)
		df = has_four_voices(df)
		#df = only_4_4(df) #reduce mucho los datos, ver que pasa
		#df = all_monophony(df)
		df = min_length(df, min_length=16)
		
	elif dataset == "ailabs":
		df = non_empty(df)
		df = non_repeated(df)
		#df = only_4_4(df) #esta mal asignado el tsc, todos son 4/4 dos veces
		#df = min_length(df, min_length=16) # no asigne n_measures

	df.reset_index(inplace=True, drop=True)
	return df


def get_clean_df(dataset_name):
	PROJECT_DIR = Path(__file__).resolve().parents[2]
	RAW_DIR = os.path.join(PROJECT_DIR, "data", "raw", dataset_name)
	FRAMES_DIR = os.path.join(PROJECT_DIR, "data", "frames")
	PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")


	df_path = os.path.join(FRAMES_DIR, dataset_name + ".pkl")
	df = pd.read_pickle(df_path)

	df = clean_df(df, dataset_name)
	indexes = df.index.values.tolist()
	random.Random(42).shuffle(indexes)
	n = len(indexes)
	train = indexes[:int(n*0.8)]
	val = indexes[len(train): len(train)+int(n*0.1)]
	test = indexes[len(train) + len(val):]

	train_set = df.loc[df.index[train]]
	val_set = df.loc[df.index[val]]
	test_set = df.loc[df.index[test]]

	train_set["set"] = train_set.apply(lambda x: "train", axis=1)
	val_set["set"] = val_set.apply(lambda x: "val", axis=1)
	test_set["set"] = test_set.apply(lambda x: "test", axis=1)

	df = pd.concat([train_set, val_set, test_set], axis=0)
	
	return df


if __name__ == '__main__':

	get_clean_df("folk")