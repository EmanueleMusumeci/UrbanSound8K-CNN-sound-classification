import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import os
import librosa

from Dataset import SoundDatasetFold
from DataLoader import DataLoader
from data_augmentation.image_transformations import *

#https://github.com/jaron/deep-listening/blob/master/1-us8k-ffn-extract-explore.ipynb
def load_sound_files(parent_dir, file_paths):
    raw_sounds = []
    for fp in file_paths:
        X,sr = librosa.load(parent_dir + fp)
        raw_sounds.append(X)
    return raw_sounds
#traduzione in PyTorch da tensorflow del tutorial:
#   https://github.com/jaron/deep-listening/blob/master/2-us8k-ffn-train-predict.ipynb

class FeedForwardNetwork(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size):
        super(FeedForwardNetwork, self).__init__()                    
        
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.relu2 = nn.ReLU()
        #1d perchè lavoro con un array
        self.drop1 = nn.Dropout(p=0.5)#aggiungi p a init variables?
        self.fc3 = nn.Linear(self.hidden_size, output_size)
        self.relu3 = nn.ReLU()
        self.drop2 = nn.Dropout(p=0.5)#aggiungi p a init variables?


    def forward(self, mfccs, chroma, mel ,contrast, tonnetz):
        x = torch.cat([mfccs, chroma, mel, contrast, tonnetz],1)
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.drop1(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.drop2(out)

        return out

