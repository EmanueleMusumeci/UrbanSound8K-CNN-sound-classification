import wave
import numpy as np

import os
import librosa

from utils import *

#https://stackoverflow.com/questions/43963982/python-change-pitch-of-wav-file
"""
import muda
import jams

"""
base_dir = os.path.dirname(os.path.realpath(__file__))
DATASET_DIR = os.path.join(base_dir,"data")
aug_dir = os.path.join(DATASET_DIR,"aug")

DATASET_NAME = os.path.join(DATASET_DIR,"UrbanSound8K")
US8K_AUDIO_PATH = os.path.join(DATASET_NAME,"audio")

fold1 = os.path.join(US8K_AUDIO_PATH,"fold1")

file_to_test = os.path.join(fold1,'7061-6-0-0.wav')

@function_timer
def numpy_test(file_to_test):
  

    wr = wave.open(file_to_test, 'r')
    # Set the parameters for the output file.
    par = list(wr.getparams())
    par[3] = 0  # The number of samples will be set by writeframes.
    par = tuple(par)
    ww = wave.open('7061-6-0-0-pitch.wav', 'w')
    ww.setparams(par)


    fr = 20
    sz = wr.getframerate()//fr  # Read and process 1/fr second at a time.
    # A larger number for fr means less reverb.
    c = int(wr.getnframes()/sz)  # count of the whole file
    shift = 100//fr  # shifting 100 Hz
    for num in range(c):


        da = np.frombuffer(wr.readframes(sz), dtype=np.int16)
        left, right = da[0::2], da[1::2]  # left and right channel
        lf, rf = np.fft.rfft(left), np.fft.rfft(right)

        lf, rf = np.roll(lf, shift), np.roll(rf, shift)
        lf[0:shift], rf[0:shift] = 0, 0

        nl, nr = np.fft.irfft(lf), np.fft.irfft(rf)
        ns = np.column_stack((nl, nr)).ravel().astype(np.int16)
        ww.writeframes(ns.tobytes())

    wr.close()
    ww.close()

#Stessa operazione in librosa
@function_timer
def librosa_test(file_to_test):
    y, sr = librosa.load(file_to_test)
    y_third = librosa.effects.pitch_shift(y, sr, n_steps=4)

if __name__=="__main__":
    numpy_test(file_to_test)
    librosa_test(file_to_test)