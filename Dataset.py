import torch

from scipy import signal

import librosa
import librosa.display

from matplotlib import pyplot as plt
import sounddevice as sd

import numpy as np
import pandas as pd

from csv import reader


def load_sound_file(file_path, sr=22050, mono=True):
    #Loads the raw sound time series and returns also the sampling rate
    raw_sound, sr = librosa.load(file_path)
    return raw_sound

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
    plt.xlabel('frequency [Hz]')
    plt.ylabel('Magnitude (norm)')

    if show:
        plt.show()
    return plot

class SoundDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, dataset_name):
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
#TODO MICHELE
        self.data = self.load_dataset(dataset_dir)
        

#TODO EMANUELE
    def __getitem__(self, index):
        #Decidere come salvare dati -> estrarre sample
        sample = self.data[index]
        class_id = sample["class_id"]
        class_name = sample["class_name"]
        meta_data = sample["meta_data"]
        
        preprocessed_sample = self.preprocess(sample)

        #Considerare quali labels usare
        return preprocessed_sample, class_id, class_name, meta_data 


    def preprocess(self, sample, spectrogram=True):
        if spectrogram:
            pass
        else:
            pass

    #lista data, ogni elemento della lista è
    #un dizionario con campi : filepath,classeId,className,
    #                           metadata= dizionario con altri dati
   
    def load_dataset(self,sample):
        file_path_audio = "UrbanSound8K-CNN-sound-classification/data/UrbanSound8K/audio/"
        with open('data/UrbanSound8K/metadata/UrbanSound8K.csv', 'r') as read_obj:
            csv_reader = reader(read_obj)
            n = 0
            audios_data_from_csv = []
            for row in csv_reader:
                audios_data_from_csv.append(row)

            list_audios = []
            for audio in audios_data_from_csv:

                metadata = {
                    "fsID":audio[1],
                    "start":audio[2],
                    "end":audio[3],
                    "salience":audio[4]
                }   
                audiodict = {
                    "file_path":file_path_audio+"fold"+audio[5]+"/"+audio[0],
                    "class_id":audio[6],
                    "class_name":audio[7],
                    "meta_data": metadata
                }

                list_audios.append(audiodict)

        return list_audios        
                
            
      

         
        
#TODO EMANUELE
        prep_sample = sample
        return prep_sample
    
    def __len__(self):
        return len(self.data)

if __name__ == "__main__":
    DATASET_DIR = "data"#data
    DATASET_NAME = "UrbanSound8K"
    dataset = SoundDataset(DATASET_DIR, DATASET_NAME)
    sound, sr = librosa.load(librosa.ex('trumpet'))
    #play_sound(sound,sr)
    #plot_sound_waves(sound, sound_file_name="file.wav", show=True, sound_class="Prova")
    #plot_sound_spectrogram(sound, sound_file_name="file.wav", show=True, sound_class="Prova", log_scale=False)
    #plot_sound_spectrogram(sound, sound_file_name="file.wav", show=True, sound_class="Prova", log_scale=True)
    #plot_sound_spectrogram(sound, sound_file_name="file.wav", show=True, sound_class="Prova", log_scale=True, title="Different hop length", hop_length=2048, sr=22050)
    #plot_periodogram(sound, sound_file_name="file.wav", show=True, sound_class="Prova")
    
    
    print(dataset.data)
   