#To produce the pickle file run only the red part commented 
#taken from : 
#           https://github.com/mariostrbac/environmental-sound-classification/blob/main/notebooks/data_preprocessing.ipynb
#           https://github.com/mariostrbac/environmental-sound-classification/blob/main/notebooks/evaluate_classifier.ipynb
"""

import os
import librosa

from tqdm import tqdm
import pandas as pd
import numpy as np

# set paths to the UrbanSound8K dataset and metadata file

base_dir = os.path.dirname(os.path.realpath(__file__))
DATASET_DIR = os.path.join(base_dir,"data")
DATASET_NAME = os.path.join(DATASET_DIR,"UrbanSound8K")
US8K_AUDIO_PATH = os.path.join(DATASET_NAME,"audio")
US8K_METADATA_PATH = os.path.join(DATASET_NAME,"metadata")
CSV = os.path.join(US8K_METADATA_PATH,"UrbanSound8K.csv")
print(CSV)
print(US8K_AUDIO_PATH)

# load the csv metadata file into a Pandas DataFrame structure
us8k_metadata_df = pd.read_csv(CSV,
                               usecols=["slice_file_name", "fold", "classID"],
                               dtype={"fold": "uint8", "classID" : "uint8"})

print(us8k_metadata_df)

HOP_LENGTH = 512        # number of samples between successive frames
WINDOW_LENGTH = 512     # length of the window in samples
N_MEL = 128             # number of Mel bands to generate


def compute_melspectrogram_with_fixed_length(audio, sampling_rate, num_of_samples=128):
    try:
        # compute a mel-scaled spectrogram
        melspectrogram = librosa.feature.melspectrogram(y=audio, 
                                                        sr=sampling_rate, 
                                                        hop_length=HOP_LENGTH,
                                                        win_length=WINDOW_LENGTH, 
                                                        n_mels=N_MEL)

        # convert a power spectrogram to decibel units (log-mel spectrogram)
        melspectrogram_db = librosa.power_to_db(melspectrogram, ref=np.max)
        
        melspectrogram_length = melspectrogram_db.shape[1]
        
        # pad or fix the length of spectrogram 
        if melspectrogram_length != num_of_samples:
            melspectrogram_db = librosa.util.fix_length(melspectrogram_db, 
                                                        size=num_of_samples, 
                                                        axis=1, 
                                                        constant_values=(0, -80.0))
    except Exception as e:
        print("\nError encountered while parsing files\n>>", e)
        return None 
    
    return melspectrogram_db


SOUND_DURATION = 2.95   # fixed duration of an audio excerpt in seconds

features = []

# iterate through all dataset examples and compute log-mel spectrograms
for index, row in tqdm(us8k_metadata_df.iterrows(), total=len(us8k_metadata_df)):
    file_path = f'{US8K_AUDIO_PATH}/fold{row["fold"]}/{row["slice_file_name"]}'
    audio, sample_rate = librosa.load(file_path, duration=SOUND_DURATION, res_type='kaiser_fast')
    
    melspectrogram = compute_melspectrogram_with_fixed_length(audio, sample_rate)
    label = row["classID"]
    fold = row["fold"]
    
    features.append([melspectrogram, label, fold])

# convert into a Pandas DataFrame 

us8k_df = pd.DataFrame(features, columns=["melspectrogram", "label", "fold"])


us8k_df.to_pickle("us8k_df.pkl")


"""

# Use this to test the net
import os
import sys
import pickle
import copy

from datetime import datetime
from tqdm import tqdm

import numpy as np
import pandas as pd
import seaborn as sns

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import matplotlib.pyplot as plt

from convolutional_model import *

us8k_df = pd.read_pickle("us8k_df.pkl")


print(us8k_df.head())


DEFAULT_SAMPLE_RATE = 22050

class UrbanSound8kDataset(Dataset):
    def __init__(self, us8k_df, transform=None):
        assert isinstance(us8k_df, pd.DataFrame)
        assert len(us8k_df.columns) == 3

        self.us8k_df = us8k_df
        self.transform = transform

    def __len__(self):
        return len(self.us8k_df)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        spectrogram, label, fold = self.us8k_df.iloc[index]

        if self.transform is not None:
            spectrogram = self.transform(spectrogram)

        return {'spectrogram': spectrogram, 'label':label}


if __name__=="__main__":
    flag_print = False
     # init train data loader
    test_us8k_df = us8k_df[us8k_df['fold'] == 1]
    test_us8k_ds = UrbanSound8kDataset(test_us8k_df)
    data_loader = DataLoader(test_us8k_ds, 
                            batch_size=16,
                            shuffle = False,
                            pin_memory=True)

    if flag_print :
        print("--------------------------------------------------------")
        print(test_us8k_df)
        print("--------------------------------------------------------")
        print(test_us8k_ds)
        print("--------------------------------------------------------")
        print(data_loader)
        print("--------------------------------------------------------")
        batch_size = torch.tensor(data_loader.batch_size)
        print("--------------------------------------------------------")
        print(batch_size)

    net = ConvolutionalNetwork(1)

    #an iterator is created (num_workers = 0)

    for step, batch in enumerate(data_loader):
        X_batch = batch['spectrogram']
        y_batch = batch['label']
        print("--------------------------------------------------------")
        print("X_batch ",X_batch.shape)
        print("--------------------------------------------------------")
        print("y_batch ",y_batch.shape)


        output = net(X_batch)
        
        print("output ",output)