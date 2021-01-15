import os

import functools
import time
import operator

import pandas as pd

from prettytable import PrettyTable

from csv import reader

import pydub
from pydub.utils import which, mediainfo
from tqdm import tqdm

import numpy as np

import sounddevice as sd
import soundfile as sf
import dill

from scipy.io import wavfile

import librosa

import matplotlib.pyplot as plt

def function_timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print("Finished {} in {} secs".format(repr(func.__name__), round(run_time, 3)))
        return value

    return wrapper

#Code block timer used as a "with" block (taken from https://stackoverflow.com/questions/30433910/decorator-to-time-specific-lines-of-the-code-instead-of-whole-method)
import contextlib

code_timer_stats = {}

@contextlib.contextmanager
def code_timer(ident):
    tstart = time.time()
    yield
    elapsed = time.time() - tstart
    print("{0}: {1} ms".format(ident, elapsed))
    if ident not in code_timer_stats:
        code_timer_stats[ident] = {"avg_time" : 0, "total_time" : 0}
    avg_time = (code_timer_stats[ident]["avg_time"] + elapsed)/2
    total_time = code_timer_stats[ident]["total_time"] + elapsed
    code_timer_stats[ident] = {"avg_time": avg_time, "total_time": total_time}

def print_code_stats():
    table = PrettyTable(['Name','Avg. time', 'Total time'])
    sort_by_row_value_index = 3
    for name, entry in code_timer_stats.items():
        row = [name]
        for _, value in entry.items():
            row.append("{:.4f}".format(value))
        table.add_row(row)
    table = table.get_string(sort_key=lambda row: row[sort_by_row_value_index], sortby="Total time", reversesort=True)
    print(table)

'''
This function loads an audio file
'''
#def load_audio_file(file_path, sample_rate=22050, mono=True):
#    #Loads the raw sound time series and returns also the sampling rate
#    raw_sound, sr = librosa.load(file_path)
#    return raw_sound

def play_sound(sound, sr = 22050, blocking=True):
    sd.play(sound, sr, blocking=True)

'''
Displays a wave plot for the input raw sound (using the Librosa library)
'''
def plot_sound_waves(sound, sound_file_name = None, sound_class=None, show=False, sr=22050):
    plot_title = "Wave plot"
    
    if sound_file_name is not None:
        plot_title += "File: "+sound_file_name
    
    if sound_class is not None:
        plot_title+=" (Class: "+sound_class+")"
    
    plot = plt.figure(plot_title)
    librosa.display.waveplot(np.array(sound),sr=sr)
    plt.title(plot_title)
    
    if show:
        plt.show()

def plot_sound_spectrogram(sound, sound_file_name = None, sound_class=None, show = False, log_scale = False, hop_length=512, sr=22050, colorbar_format = "%+2.f dB", title=None):
    if title is None:
        plot_title = title
    else:
        plot_title = "Spectrogram"
        
        if sound_file_name is not None:
            plot_title += "File: "+sound_file_name
        
        if sound_class is not None:
            plot_title+=" (Class: "+sound_class+")"
    
    sound = librosa.stft(sound, hop_length = hop_length)
    sound = librosa.amplitude_to_db(np.abs(sound), ref=np.max)

    if log_scale:
        y_axis = "log"
    else:
        y_axis = "linear"

    plot = plt.figure(plot_title)
    librosa.display.specshow(sound, hop_length = hop_length, x_axis="time", y_axis=y_axis)

    plt.title(plot_title)
    plt.colorbar(format=colorbar_format)
    
    if show:
        plt.show()
    
    return plot

#from https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.periodogram.html
def plot_periodogram(sound, sound_file_name = None, sound_class=None, show = False, sr=22050, title=None):
    f, Pxx_den = signal.periodogram(sound, sr)
    plot = plt.figure()
    plt.semilogy(f, Pxx_den)
    plt.ylim([1e-7, 1e2])
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude (norm)')

    if show:
        plt.show()
    return plot
    
def display_heatmap(data):
    plt.imshow(data, cmap="hot", interpolation='nearest')
    plt.show()

def unpickle_data(path):
    with open(path, "rb") as f:
      data = dill.load(f)
      return data

def pickle_data(data, path):
    with open(path, "wb+") as f:
        dill.dump(data, f)

#@function_timer
#from https://github.com/karolpiczak/paper-2015-esc-convnet/blob/master/Code/_Datasets/Setup.ipynb
def load_compacted_dataset(dataset_dir, folds = []):
    """Load raw audio and metadata content from the UrbanSound8K dataset."""
    
    audio_meta = []
    audio_raw = []
    for fold in folds:
        if os.path.isfile(os.path.join(dataset_dir,'urban_meta_fold_{}.pkl'.format(fold))) and os.path.isfile(os.path.join(dataset_dir,'urban_audio_fold_{}.dat'.format(fold))):
            fold_meta = unpickle_data(os.path.join(dataset_dir,'urban_meta_fold_{}.pkl'.format(fold)))
            fold_raw = np.memmap(os.path.join(dataset_dir,'urban_audio_fold_{}.dat'.format(fold)), dtype='float32', mode='r', shape=(len(fold_meta), 88200))
            audio_meta = np.concatenate((audio_meta,fold_meta))
            audio_raw.append(fold_raw)
        else:
            raise FileNotFoundError

    audio_raw = np.vstack(audio_raw)
    return audio_meta, audio_raw

#Snippet taken from https://github.com/karolpiczak/paper-2015-esc-convnet/blob/master/Code/_Datasets/Setup.ipynb
'''
This function loads an audio file for a specified duration (default 4 secs), overlays on a silent audio segment 
(providing a native padding), resamples it a t 22050 Hz, sets it to mono and converts it to float
'''
#@function_timer
def _load_audio_file(path, duration = 4000, sample_rate = 22050):

    print(mediainfo(path)["sample_rate"])
    print(mediainfo(path))
    
    #with code_timer("pydub.AudioSegment.silent"):
    silence = pydub.AudioSegment.silent(duration=duration, frame_rate=sample_rate)
    
    try:
        #with code_timer("pydub.AudioSegment.from_file"):
        audio_segment = pydub.AudioSegment.from_file(path).set_frame_rate(sample_rate).set_channels(1)
    #Small fix for the infamous pydub encoder exception with ffmpeg decoder (https://github.com/jiaaro/pydub/issues/415)
    except pydub.exceptions.CouldntDecodeError:
        audio_segment = pydub.AudioSegment.from_file_using_temporary_files(path).set_frame_rate(sample_rate).set_channels(1)

    #with code_timer("audio.overlay"):
    #print(len(audio_segment._data))
    #print(len(silence._data))
    audio_segment = silence.overlay(audio_segment)
    #print(len(audio_segment._data))
    #print("\n")
    
    #with code_timer("np.from_buffer"):
    raw = (np.frombuffer(audio_segment._data, dtype="int16") + 0.5) / (0x7FFF + 0.5)   # convert to float
    if audio_segment.duration_seconds != duration/1000 or len(raw) != duration/1000 * sample_rate:
        print("Audio segment {} has duration {} and array length {}".format(path, audio_segment.duration_seconds,len(raw)))

    return raw, sample_rate

def load_audio_file(path, duration = 4000, sample_rate = 22050, fixed_length = 88200):
    '''
    samplerate, data = wavfile.read(path)
    print("scipy")
    print(str(data)[:100])
    print(len(data))
    print(sample_rate)
    audio_segment = pydub.AudioSegment.from_file(path)
    print("pydub")
    print(str(audio_segment._data)[:100])
    print(len(audio_segment._data))
    print(sample_rate)
    data, samplerate = sf.read(path)
    print("soundfile")
    print(str(data)[:100])
    print(len(data))
    print(sample_rate)
    '''
    data, sample_rate = librosa.load(path, sr=sample_rate, mono=True,  dtype=np.float32)
    data = np.concatenate((data, np.zeros(int((fixed_length - len(data))))))
    #print(data.shape)
    #if len(data)>88200:
    #    print("librosa")
    #    print(str(data)[:100])
    #    print(len(data))
    #    print(sample_rate)
    #    print(data)
        #play_sound(data)
    return data, sample_rate

#from https://github.com/karolpiczak/paper-2015-esc-convnet/blob/master/Code/_Datasets/Setup.ipynb
def compact_urbansound_dataset(dataset_dir, folds = [1,2,3,4,5,6,7,8,9,10], skip_first_line = True):
    """Load raw audio and metadata content from the UrbanSound8K dataset and generate a .dat file"""
    total_files = 0
    for fold in folds:
        total_files += len([name for name in os.listdir(os.path.join(dataset_dir, "UrbanSound8K", "audio", "fold"+str(fold))) if os.path.isfile(os.path.join(dataset_dir, "UrbanSound8K", "audio", "fold"+str(fold),name))])
    
    with open(os.path.join(dataset_dir,'UrbanSound8K','metadata','UrbanSound8K.csv'), 'r') as read_obj:
        csv_reader = reader(read_obj)
        
        #next skips the first line that contains the header info
        if skip_first_line: 
            next(csv_reader)

        audios_data_from_csv = []
        for row in csv_reader:
            audios_data_from_csv.append(row)

        audio_ids = []
        audio_meta = {}
        audio_raw = {}

        print("Compacting {} folds ({} files)...".format(len(folds), total_files))
        progress_bar = tqdm(total=total_files, desc="Sample", position=0)
        for i, audio in enumerate(audios_data_from_csv):
            fold_number = int(audio[5])
            if fold_number in folds:
                
                metadata = {
                    "fsID":audio[1],
                    "start":audio[2],
                    "end":audio[3],
                    "salience":audio[4]
                }   
                audiodict = {
                    "file_path": os.path.join(dataset_dir, "UrbanSound8K", "audio", "fold"+str(fold_number), audio[0]),
                    "class_id":audio[6],
                    "class_name":audio[7],
                    "meta_data": metadata,
                }
                
                if fold_number not in audio_meta.keys():
                    audio_meta[fold_number] = []    
                    audio_raw[fold_number] = []

                audio_file, sr = load_audio_file(audiodict["file_path"], 4000)
                #progress_bar.update(1)

                #continue
                #print(len(audio_file))
                audiodict["sampling_rate"] = sr
                audio_meta[fold_number].append(audiodict)
                audio_raw[fold_number].append(audio_file)
                progress_bar.update(1)

    print("Saving meta-data...")
    for fold, meta_data in audio_meta.items():
        pickle_data(meta_data, os.path.join(dataset_dir,"urban_meta_fold_{}.pkl".format(fold)))


    print("Compacting raw audio...")
    for fold, audio_clips in audio_raw.items():
        audio_raw_stacked = np.vstack(audio_clips)
        mm = np.memmap(os.path.join(dataset_dir,'urban_audio_fold_{}.dat'.format(fold)), dtype='float32', mode='w+', shape=(len(audio_raw_stacked), 88200))
        mm[:] = audio_raw_stacked[:]
    mm.flush()
    del audio_raw
    return audio_meta, mm

if __name__=="__main__":

    base_dir = os.path.dirname(os.path.realpath(__file__))
    DATASET_DIR = os.path.join(base_dir,"data")
    audio_meta, mm = compact_urbansound_dataset(DATASET_DIR,folds = [2])
    #audio_meta, audio_raw = load_compacted_dataset(DATASET_DIR,folds = [2,3,4,5,6,7,8,9,10])

    for i, audio in enumerate(audio_raw):
        print("{}: {}".format(i,audio))
