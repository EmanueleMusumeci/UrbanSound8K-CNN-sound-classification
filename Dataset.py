import random

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
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude (norm)')

    if show:
        plt.show()
    return plot
    


class SoundDatasetFold(torch.utils.data.IterableDataset):
    def __init__(self, dataset_dir, dataset_name, 
                folds = [], 
                shuffle_dataset = False, 
                generate_spectrograms = True, 
                image_augmentation_pipeline = None,
                test = False
                ):
        super(SoundDatasetFold).__init__()
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        self.folds = folds
        self.data, self.data_ids = self.load_dataset(dataset_dir, folds=self.folds) 
        self.generate_spectrograms = generate_spectrograms
        self.image_augmentation_pipeline = image_augmentation_pipeline
    
    def get_preprocessed_fields(): 
        if self.generate_spectrograms:
            return ["original_spectrogram", "preprocessed_spectrogram"]
        else:
            return ["mfccs", "chroma", "mel", "contrast", "tonnetz"]

    def get_unpreprocessed_fields(): return ["class_id", "class_name", "meta_data"]

    def __call__(self, sound, sample_rate=22050):
        if self.generate_spectrograms:
            original_spectrograms, preprocessed_spectrograms = self.preprocess(sound, sample_rate=sample_rate, spectrograms=True)
            returned_samples = []
            for orig_spec, prep_spec in zip(original_spectrograms, preprocessed_spectrograms):

                returned_samples.append({
                        "original_spectrogram" : orig_spec, 
                        "preprocessed_spectrogram" : prep_spec, 
                        "class_id" : None, 
                        "class_name" : None, 
                        "meta_data" : None
                        })
            return returned_samples
        else:
            mfccs, chroma, mel, contrast, tonnetz = self.preprocess(sample, sample_rate=sample_rate, spectrograms=False)
            return [{
                    "mfccs" : mfccs, 
                    "chroma" : chroma, 
                    "mel" : mel, 
                    "contrast" : contrast, 
                    "tonnetz" : tonnetz, 
                    "class_id" : None, 
                    "class_name" : None, 
                    "meta_data" : None
                    }]


    def __getitem__(self, index):
        sample = self.data[index]
        class_id = sample["class_id"]
        class_name = sample["class_name"]
        meta_data = sample["meta_data"]
        
        sound, sample_rate = librosa.load(sample["file_path"])  

        if self.generate_spectrograms:
            original_spectrograms, preprocessed_spectrograms = self.preprocess(sound, sample_rate = sample_rate, spectrograms=True)
            returned_samples = []
            for orig_spec, prep_spec in zip(original_spectrograms, preprocessed_spectrograms):

                returned_samples.append({
                        "original_spectrogram" : orig_spec, 
                        "preprocessed_spectrogram" : prep_spec, 
                        "class_id" : class_id, 
                        "class_name" : class_name, 
                        "meta_data" : meta_data
                        })
            return returned_samples
        else:
            mfccs, chroma, mel, contrast, tonnetz = self.preprocess(sound, sample_rate = sample_rate, spectrograms=False)
            return [{
                    "mfccs" : mfccs, 
                    "chroma" : chroma, 
                    "mel" : mel, 
                    "contrast" : contrast, 
                    "tonnetz" : tonnetz, 
                    "class_ids" : class_id, 
                    "class_names" : class_names, 
                    "meta_data" : meta_data
                    }]

    
    #pattern iter, chiamo n=len(dataset) volte __getitem__ (nostro metodo getter)
    #con yield
    #forse spacchetto dati di __getitem__, e.q se due finestre, una alla volta
    def __iter__(self):
        if shuffle_dataset:
            random.shuffle(image_ids)

        for index in self.data_ids:
            preprocessed_samples, class_id, class_name, meta_data = self[index]
            for sample in preprocessed_samples:
                yield sample
                
#TODO implement the FFN data preprocessing
    def preprocess(self, sound, sample_rate = 22050, spectrogram=True):  
        if not spectrogram:
            #extract_feature
            print("Features :"+len(sound)+" sampled at "+sample_rate+"hz")
            #Short-time Fourier transform(STFT)
            stft = np.abs(librosa.stft(sound))
            #Mel-frequency cepstral coefficients (MFCCs)
            mfccs = np.mean(librosa.feature.mfcc(y=sound, sr=sample_rate, n_mfcc=40).T,axis=0)
            #Compute a chromagram from a waveform or power spectrogram.
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            #Compute a mel-scaled spectrogram.
            mel = np.mean(librosa.feature.melspectrogram(sound, sr=sample_rate).T,axis=0)
            contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
            #Computes the tonal centroid features (tonnetz)
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(sound), sr=sample_rate).T,axis=0)
            
            return mfccs, chroma, mel, contrast, tonnetz
        else:
            pass

    #lista data, ogni elemento della lista è
    #un dizionario con campi : filepath,classeId,className,
    #                           metadata= dizionario con altri dati
   
    def load_dataset(self,sample, folds = []):
        file_path_audio = "UrbanSound8K-CNN-sound-classification/data/UrbanSound8K/audio/"
        with open('data/UrbanSound8K/metadata/UrbanSound8K.csv', 'r') as read_obj:
            csv_reader = reader(read_obj)
            audios_data_from_csv = []
            for row in csv_reader:
                audios_data_from_csv.append(row)

            index = 0
            list_audios = {}
            for audio in audios_data_from_csv:
                fold = audio[5]
                if fold not in folds:
                    metadata = {
                        "fsID":audio[1],
                        "start":audio[2],
                        "end":audio[3],
                        "salience":audio[4]
                    }   
                    audiodict = {
                        "file_path":file_path_audio+"fold"+fold+"/"+audio[0],
                        "class_id":audio[6],
                        "class_name":audio[7],
                        "meta_data": metadata
                    }

                    list_audios[index] = audiodict
                    audio_ids = index
                    index += 1

        return list_audios, audio_ids     
            
    def __len__(self):
        return len(self.data)

if __name__ == "__main__":
    DATASET_DIR = "data"#data
    DATASET_NAME = "UrbanSound8K"
    dataset = SoundDatasetFold(DATASET_DIR, DATASET_NAME, generate_spectrograms=False)
    sound, sr = librosa.load(librosa.ex('trumpet'))
    #play_sound(sound,sr)
    #plot_sound_waves(sound, sound_file_name="file.wav", show=True, sound_class="Prova")
    #plot_sound_spectrogram(sound, sound_file_name="file.wav", show=True, sound_class="Prova", log_scale=False)
    #plot_sound_spectrogram(sound, sound_file_name="file.wav", show=True, sound_class="Prova", log_scale=True)
    #plot_sound_spectrogram(sound, sound_file_name="file.wav", show=True, sound_class="Prova", log_scale=True, title="Different hop length", hop_length=2048, sr=22050)
    #plot_periodogram(sound, sound_file_name="file.wav", show=True, sound_class="Prova")
    
    
    #print(dataset.data)

    mfccs,chroma,mel,contrast,tonnetz = preprocess(dataset.data[0],False)

    print("mfcss : "+mfcss)
    print("chroma: "+chroma)
    print("mel: "+ mel)
    print("contrast: "+contrast)
    print("tonnetz: "+tonnetz)
   