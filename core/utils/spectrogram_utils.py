from enum import Enum
import math

import numpy as np

import librosa
from matplotlib import pyplot as plt
import scipy
from scipy import signal

from core.utils.timing import *

'''
Display a matrix as a heatmap
'''
def display_heatmap(data):
    plt.imshow(data, cmap="hot", interpolation='nearest')
    plt.show()

'''
Compute the number of frames in a log mel spectrogram according to its duration (seconds),
sample rate and hop length
'''
def compute_spectrogram_frames(clip_seconds, sample_rate, hop_length):
    return math.ceil((clip_seconds*sample_rate)/hop_length)

'''
Generates a (log) mel spectrogram using the librosa library
'''
def generate_mel_spectrogram_librosa(signal, 
                                    spectrogram_bands=128, hop_length=512, 
                                    sample_rate=22050, log_mel=True, debug_time_label="", 
                                    show=False, flatten=False):

    debug_time = len(debug_time_label)>0

    #Generate log-mel spectrogram spectrogram of preprocessed signal segment
    with code_timer(debug_time_label+"librosa.feature.melspectrogram", debug = debug_time):
        mel_spectrogram = librosa.feature.melspectrogram(signal, n_mels = spectrogram_bands, hop_length=hop_length, sr=sample_rate)

    if log_mel:
        with code_timer(debug_time_label+"librosa.amplitude_to_db", debug = debug_time):
            mel_spectrogram = librosa.amplitude_to_db(mel_spectrogram)

    if show:
        display_heatmap(mel_spectrogram)

    if flatten:
        with code_timer(debug_time_label+"mel_spectrogram.T.flatten()", debug = debug_time):
            mel_spectrogram = mel_spectrogram.T.flatten()[:, np.newaxis].T

    return mel_spectrogram
