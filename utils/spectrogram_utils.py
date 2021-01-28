from enum import Enum
import math

import numpy as np

import librosa
from matplotlib import pyplot as plt
import scipy
from scipy import signal

try:
    from utils.timing import *
except:
    pass

def display_heatmap(data):
    plt.imshow(data, cmap="hot", interpolation='nearest')
    plt.show()

def compute_spectrogram_frames(clip_seconds, sample_rate, hop_length):
    return math.ceil((clip_seconds*sample_rate)/hop_length)

def generate_mel_spectrogram_librosa(signal, spectrogram_bands, log_mel=True, debug_time_label="", show=False, flatten=False):

    debug_time = len(debug_time_label)>0

    #Generate log-mel spectrogram spectrogram of preprocessed signal segment
    with code_timer(debug_time_label+"librosa.feature.melspectrogram", debug = debug_time):
        mel_spectrogram = librosa.feature.melspectrogram(signal, n_mels = spectrogram_bands)

    if log_mel:
        with code_timer(debug_time_label+"librosa.amplitude_to_db", debug = debug_time):
            mel_spectrogram = librosa.amplitude_to_db(mel_spectrogram)

    if show:
        display_heatmap(mel_spectrogram)

    if flatten:
        with code_timer(debug_time_label+"mel_spectrogram.T.flatten()", debug = debug_time):
            mel_spectrogram = mel_spectrogram.T.flatten()[:, np.newaxis].T

    return mel_spectrogram

'''
from essentia.standard import *
essentia_windowing_algo = Windowing(type = 'hann')
essentia_spectrum_algo = Spectrum()  # FFT() would return the complex FFT, here we just want the magnitude spectrum
essentia_log_spectrum_algo = LogSpectrum()  # FFT() would return the complex FFT, here we just want the magnitude spectrum
essentia_mfcc_algo = MFCC()

def generate_mel_spectrogram_essentia(signal, spectrogram_bands, sample_rate, log_mel=True, debug_time_label=""):

    debug_time = len(debug_time_label)>0

    if log_mel:
        with code_timer(debug_time_label+"essentia.LogSpectrum", debug = debug_time):
            mel_spectrogram, _, _ = essentia_log_spectrum_algo(signal, sampleRate = sample_rate)
            essentia_log_spectrum_algo.reset()
    else:
        with code_timer(debug_time_label+"essentia.Spectrum", debug = debug_time):
            mel_spectrogram = essentia_spectrum_algo(signal)
            essentia_spectrum_algo.reset()

    with code_timer(debug_time_label+"essentia.flatten", debug = debug_time):
        mel_spectrogram = essentia.array(mel_spectrogram).flatten()[:, np.newaxis].T

    return mel_spectrogram


def generate_ffn_features_essentia(signal, sample_rate, bands = 60, coefficients = 1):
    spectrum = essentia_spectrum_algo(signal)
    essentia_spectrum_algo.reset()

    bands, mfcc = essentia_mfcc_algo(spectrum, sampleRate=sample_rate, numberBands = bands, numberCoefficients = coefficients)
    mfcc = essentia.array(mfcc).T

    return bands, mfcc
'''