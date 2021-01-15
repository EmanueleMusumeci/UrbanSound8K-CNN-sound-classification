import os

import random
import math

import pydub
from pydub.utils import which
#This should fix the ffmpeg decoding errors as in https://github.com/jiaaro/pydub/issues/173
pydub.AudioSegment.converter = which("ffmpeg")

#import warnings
#warnings.filterwarnings("once", category=UserWarning)

import torch, torchvision

from torchvision import transforms

from scipy import signal

import librosa
import librosa.display

from matplotlib import pyplot as plt
import sounddevice as sd

import numpy as np
import pandas as pd

from csv import reader

from tqdm import tqdm

#from DataLoader import DataLoader

from image_transformations import SpectrogramAddGaussNoise, SpectrogramReshape, SpectrogramShift

from utils import function_timer, code_timer, print_code_stats, display_heatmap, play_sound, load_audio_file, \
    pickle_data, unpickle_data, \
    load_compacted_dataset, compact_urbansound_dataset

class SoundDatasetFold(torch.utils.data.IterableDataset):
    def __init__(self, dataset_dir, dataset_name, 
                folds = [], 
                shuffle_dataset = False, 
                generate_spectrograms = True, 

                shift_transformation = None,
                background_noise_transformation = None,

                audio_augmentation_pipeline = [],

                audio_clip_duration = 4000,
                sample_rate = 22050,

                spectrogram_bands = 60,
                spectrogram_frames_per_segment = 41,
                spectrogram_window_overlap = 0.5,
                drop_last_spectrogram_frame = True,
                normalize_audio = True,
                silent_clip_cutoff_dB = -70,

                compute_deltas = True,
                compute_delta_deltas = False,
                test = False,
                progress_bar = False,
                debug_preprocessing = False,

                load_compacted = True
                ):
        super(SoundDatasetFold).__init__()
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        self.folds = folds

        self.load_compacted = load_compacted
        if self.load_compacted:
            self.audio_meta, self.audio_raw = load_compacted_dataset(dataset_dir, folds=self.folds) 
            self.sample_ids = np.arange(0, len(self.audio_meta))
        else:
            self.audio_meta, self.sample_ids = self.load_dataset_index(dataset_dir, folds=self.folds) 

        self.shuffle_dataset = shuffle_dataset

        self.generate_spectrograms = generate_spectrograms

        self.shift_transformation = shift_transformation
        self.background_noise_transformation = background_noise_transformation

        self.audio_augmentation_pipeline = audio_augmentation_pipeline

        self.audio_clip_duration = audio_clip_duration
        self.sample_rate = sample_rate
        
        #as in https://github.com/karolpiczak/paper-2015-esc-convnet/blob/master/Code/_Datasets/Setup.ipynb
        self.spectrogram_frames_per_segment = spectrogram_frames_per_segment
        self.spectrogram_window_size = 512 * (spectrogram_frames_per_segment-1)
        assert spectrogram_window_overlap > 0 and spectrogram_window_overlap < 1, "spectrogram_window_overlap should be between 0 and 1"
        self.spectrogram_window_overlap = spectrogram_window_overlap
        self.spectrogram_window_step_size = math.floor(self.spectrogram_window_size * (1-spectrogram_window_overlap))
        self.spectrogram_bands = spectrogram_bands
        self.drop_last_spectrogram_frame = drop_last_spectrogram_frame

        self.silent_clip_cutoff_dB = silent_clip_cutoff_dB

        self.normalize_audio = normalize_audio

        self.compute_deltas = compute_deltas
        self.compute_delta_deltas = compute_delta_deltas
    
        self.progress_bar = progress_bar

        self.debug_preprocessing = debug_preprocessing

    def get_preprocessed_fields(self): 
        if self.generate_spectrograms:
            return ["original_spectrogram", "preprocessed_spectrogram"]
        else:
            return ["mfccs", "chroma", "mel", "contrast", "tonnetz"]

    def get_unpreprocessed_fields(self): return ["class_id", "class_name", "meta_data"]
    def get_gold_fields(self): return []

    def __call__(self, sound, sample_rate=22050):
        if self.generate_spectrograms:
            print("call-> if")
            original_spectrograms, preprocessed_spectrograms = self.preprocess(sound,spectrograms=True)
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
            print("call else: ")
            print("sound: ",sound)
            mfccs, chroma, mel, contrast, tonnetz = self.preprocess(sound,spectrogram=False)
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

    #@function_timer
    def __getitem__(self, index):
        debug = False
        #Decidere come salvare dati -> estrarre sample
        sample = self.audio_meta[index]

        class_id = sample["class_id"]
        class_name = sample["class_name"]
        meta_data = sample["meta_data"]
        
        if debug: print(sample["file_path"], end="")
        if self.load_compacted:
            sound = self.audio_raw[index]
        else:
            try:
                #sound, sample_rate = librosa.load(sample["file_path"])  
                sound, sample_rate = load_audio_file(sample["file_path"], sample_rate=self.sample_rate)
                #print("sound: ",sound)
            except pydub.exceptions.CouldntDecodeError as e:
                #print("EXCEPTION")
                raise e

        if self.generate_spectrograms:
            print("entrato in if")
            original_spectrograms, preprocessed_spectrograms = self.preprocess(sound, spectrogram=True)
            returned_samples = []
            for orig_spec, prep_spec in zip(original_spectrograms, preprocessed_spectrograms):

                returned_samples.append({
                        "original_spectrogram" : orig_spec, 
                        "preprocessed_spectrogram" : prep_spec, 
                        "class_id" : class_id, 
                        "class_name" : class_name, 
                        "meta_data" : meta_data
                        })
            
            if debug: print(" Returned samples: "+str(len(returned_samples)))
            return returned_samples
        else:
            print("getitem -> entrato in else:")
            print(sound)
            mfccs, chroma, mel, contrast, tonnetz = self.preprocess(sound, spectrogram=False)
            if debug: print(" Returned samples: "+str(1))
            return [{
                    "mfccs" : mfccs, 
                    "chroma" : chroma, 
                    "mel" : mel, 
                    "contrast" : contrast, 
                    "tonnetz" : tonnetz, 
                    "class_ids" : class_id, 
                    "class_names" : class_name, 
                    "meta_data" : meta_data
                    }]
    
    #pattern iter, chiamo n=len(dataset) volte __getitem__ (nostro metodo getter)
    #con yield
    #forse spacchetto dati di __getitem__, e.q se due finestre, una alla volta
    def __iter__(self):
        if self.shuffle_dataset:
            random.shuffle(self.sample_ids)

        if self.progress_bar:
            progress_bar = tqdm(total=len(self.sample_ids), desc="Sample", position=0)
        for index in self.sample_ids:
            try:
                preprocessed_samples = self[index]
                previous_samples = preprocessed_samples
            #This is a temporary fix to a decoding error with files from clip 36429: we provide the last sample twice 
            except pydub.exceptions.CouldntDecodeError:
                preprocessed_samples = previous_samples

            if self.progress_bar:
                progress_bar.update(1)
            for sample in preprocessed_samples:
                yield sample

        if self.progress_bar:
            progress_bar.close()
                
#TODO implement the FFN data preprocessing
    #@function_timer
    def preprocess(self, audio_clip, spectrogram=True, debug = False):
        print(audio_clip)
        def overlapping_segments_generator(step_size, window_size, total_frames, drop_last = True):

            start = 0
            while start < total_frames:
                yield start, start + window_size
                start += step_size
            #If true, the last segment will be dropped if its length is lower than the segment size
            if not drop_last:
                yield start, total_frames-1
        """
        if self.normalize_audio:
            print("normalize -> audio_clip: ", audio_clip)
            normalization_factor = 1 / np.max(np.abs(audio_clip)) 
            audio_clip = audio_clip * normalization_factor
        """

        if not spectrogram:
            #extract_feature
            if debug: print("Features :"+str(len(audio_clip))+"sampled at "+str(sample_rate)+"hz")
            #Short-time Fourier transform(STFT)
            print("preprocess -> audio_clip: ",audio_clip)

            stft = np.abs(librosa.stft(audio_clip))
            if debug: print("stft:\n"+str(stft))
            #Mel-frequency cepstral coefficients (MFCCs)
            if debug: print("before mfccs:\n"+str(stft))
            sample_rate = self.sample_rate
            mfccs = np.mean(librosa.feature.mfcc(S=audio_clip, sr=sample_rate, n_mfcc=40).T,axis=0)
            if debug: print("mfccs:\n"+str(mfccs))

            #Compute a chromagram from a waveform or power spectrogram.
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            if debug: print("chroma:\n"+str(chroma))

            #Compute a mel-scaled spectrogram.
            mel = np.mean(librosa.feature.melspectrogram(audio_clip, sr=sample_rate).T,axis=0)
            if debug: print("mel:\n"+str(mel))

            contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
            if debug: print("contrast:\n"+str(contrast))

            #The warning is triggered by this problem: https://github.com/librosa/librosa/issues/1214
            #Computes the tonal centroid features (tonnetz)
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(audio_clip), sr=sample_rate).T,axis=0)
            if debug: print("tonnetz:\n"+str(tonnetz))
            
            return mfccs, chroma, mel, contrast, tonnetz
        else:
            original_spectrograms = []
            preprocessed_spectrograms = []
            if debug: print("segments: "+str(list(overlapping_segments_generator(self.spectrogram_window_step_size, self.spectrogram_window_size, len(audio_clip), self.drop_last_spectrogram_frame))))
            for i, (segment_start, segment_end) in enumerate(overlapping_segments_generator(self.spectrogram_window_step_size, self.spectrogram_window_size, len(audio_clip), self.drop_last_spectrogram_frame)):
                original_signal_segment = audio_clip[segment_start:segment_end]
                #Only accept audio clip segments that are:
                #1) Of a fixed window size (self.spectrogram_window_size)
                #2) Fully contained in the audio clip (segments that go "out of bounds" wrt the audio clip are not considered)
                if len(original_signal_segment) == math.floor(self.spectrogram_window_size) and segment_end<len(audio_clip): 
                    if debug: print("segment ("+str(segment_start)+", "+str(segment_end)+")")

                    with code_timer("original librosa.feature.melspectrogram"):
                        #Generate log-mel spectrogram spectrogram of original signal segment
                        original_mel_spectrogram = librosa.feature.melspectrogram(original_signal_segment, n_mels = self.spectrogram_bands)
                    with code_timer("original librosa.amplitude_to_db"):
                        original_log_mel_spectrogram = librosa.amplitude_to_db(original_mel_spectrogram)
                    with code_timer("original original_log_mel_spectrogram.T.flatten()"):
                        original_log_mel_spectrogram = original_log_mel_spectrogram.T.flatten()[:, np.newaxis].T
                    
                    with code_timer("drop silent"):
                        #drop silent frames (taken from https://github.com/karolpiczak/paper-2015-esc-convnet/blob/master/Code/_Datasets/Setup.ipynb)
                        #ONLY if they aren't the only frame of the audio clip
                        if debug: print("Mean dB value:" +str(np.mean(original_log_mel_spectrogram)))
                        if i>0 and np.mean(original_log_mel_spectrogram) <= self.silent_clip_cutoff_dB:
                            if debug: print("Silent segment dropped")
                            continue
                    
                    original_spectrograms.append(original_log_mel_spectrogram)

                    #Apply all audio augmentations, in sequence
                    preprocessed_signal_segment = original_signal_segment

                    with code_timer("audio augmentation"):
    #TODO Try to use transforms.Compose also for audio segments (might work if we respect the class structure!)
                        for augmentation in self.audio_augmentation_pipeline:
                            preprocessed_signal_segment = augmentation(preprocessed_signal_segment)

                    #Generate log-mel spectrogram spectrogram of preprocessed signal segment
                    with code_timer("preprocessed librosa.feature.melspectrogram"):
                        preprocessed_mel_spectrogram = librosa.feature.melspectrogram(preprocessed_signal_segment, n_mels = self.spectrogram_bands)
                    with code_timer("preprocessed librosa.amplitude_to_db"):
                        preprocessed_log_mel_spectrogram = librosa.amplitude_to_db(preprocessed_mel_spectrogram)
                    with code_timer("preprocessed original_log_mel_spectrogram.T.flatten()"):
                        preprocessed_log_mel_spectrogram = preprocessed_log_mel_spectrogram.T.flatten()[:, np.newaxis].T
                                            
                        preprocessed_spectrograms.append(preprocessed_log_mel_spectrogram)
            
            with code_timer("original spectrogram reshape"):
            #Reshape the spectrograms from [N_AUDIO_SEGMENT, N_BANDS, N_FRAMES] to [N_AUDIO_SEGMENT, N_BANDS, N_FRAMES, 1]
                original_spectrograms = np.asarray(original_spectrograms).reshape(len(original_spectrograms),self.spectrogram_bands,self.spectrogram_frames_per_segment,1)
            
            with code_timer("preprocessed spectrogram reshape"):
                preprocessed_spectrograms = np.asarray(preprocessed_spectrograms).reshape(len(preprocessed_spectrograms),self.spectrogram_bands,self.spectrogram_frames_per_segment,1)
            
            with code_timer("deltas"):
                if self.compute_deltas:
                    #Make space for the delta features
                    preprocessed_spectrograms_with_deltas = np.concatenate((preprocessed_spectrograms, np.zeros(np.shape(preprocessed_spectrograms))), axis = 3)
                    if self.compute_delta_deltas:
                        preprocessed_spectrograms_with_deltas = np.concatenate((preprocessed_spectrograms_with_deltas, np.zeros(np.shape(preprocessed_spectrograms))), axis = 3)


                    for i in range(len(preprocessed_spectrograms_with_deltas)):
                        preprocessed_spectrograms_with_deltas[i, :, :, 1] = librosa.feature.delta(preprocessed_spectrograms_with_deltas[i, :, :, 0])
                        
                        if self.compute_delta_deltas:
                            preprocessed_spectrograms_with_deltas[i, :, :, 2] = librosa.feature.delta(preprocessed_spectrograms_with_deltas[i, :, :, 1])

            #preprocessed_spectrograms_with_deltas is the preprocessed output with shape [N_AUDIO_SEGMENTS, N_BANDS, N_FRAMES, N_FEATURES] where
            #N_FEATURES is 1, 2 or 3 depending on our choice of computing delta and delta-delta spectrograms


            with code_timer("image augmentation"):
#TODO Testare image transformations
                #Spectrogram image transformation
                for i in range(len(preprocessed_spectrograms_with_deltas)):
                    #Background noise transformations should not be applied to deltas
                    if self.background_noise_transformation is not None:
                        preprocessed_spectrogram, noise_mask = self.background_noise_transformation(preprocessed_spectrograms_with_deltas[i, :, :, 0])
                        preprocessed_spectrograms_with_deltas[i, :, :, 0] = preprocessed_spectrogram

                    if self.shift_transformation is not None:
                        shifted_spectrogram, shift_position = self.shift_transformation(preprocessed_spectrograms_with_deltas[i, :, :, 0])
                        preprocessed_spectrograms_with_deltas[i, :, :, 0] = shifted_spectrogram

                        #Spectrogram shift transformations can be applied to delta and delta-delta histograms as well
                        if self.compute_deltas:
                            shifted_delta_spectrogram, _ = self.shift_transformation(preprocessed_spectrograms_with_deltas[i, :, :, 1], shift_position=shift_position)
                            preprocessed_spectrograms_with_deltas[i, :, :, 1] = shifted_delta_spectrogram
                            if self.compute_delta_deltas:
                                shifted_delta_delta_spectrogram, _ = self.shift_transformation(preprocessed_spectrograms_with_deltas[i, :, :, 2], shift_position=shift_position)
                                preprocessed_spectrograms_with_deltas[i, :, :, 2] = shifted_delta_delta_spectrogram
                
            return original_spectrograms, preprocessed_spectrograms_with_deltas

    #lista data, ogni elemento della lista è
    #un dizionario con campi : filepath,classeId,className,
    #                           metadata= dizionario con altri dati
    def load_dataset_index(self, sample, folds = [], skip_first_line=True):
        with open(os.path.join(self.dataset_dir,'metadata','UrbanSound8K.csv'), 'r') as read_obj:
            csv_reader = reader(read_obj)
            
            #next skips the first line that contains the header info
            if skip_first_line: next(csv_reader)

            audios_data_from_csv = []
            for row in csv_reader:
                audios_data_from_csv.append(row)

            index = 0
            audio_ids = []
            audio_meta = []
            for audio in audios_data_from_csv:
                fold_number = audio[5]
                if fold_number not in self.folds:
                    
                    metadata = {
                        "fsID":audio[1],
                        "start":audio[2],
                        "end":audio[3],
                        "salience":audio[4]
                    }   
                    audiodict = {
                        "file_path": os.path.join(self.dataset_dir, self.dataset_name, "audio", "fold"+str(fold_number), audio[0]),
                        "class_id":audio[6],
                        "class_name":audio[7],
                        "meta_data": metadata
                    }

                    audio_meta.append(audiodict)
                    audio_ids.append(index)
                    index += 1
        return audio_meta, audio_ids     
            
    #def __len__(self):
    #    #if self.generate_spectrograms:
    #    #    return len(self.data) * self.fixed_segment_size
    #    #else:
    #    return len(self.data)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.realpath(__file__))
    DATASET_DIR = os.path.join(base_dir,"data")
    DATASET_NAME = "UrbanSound8K"
    
    spectrogram_frames_per_segment = 41
    spectrogram_bands = 60

    CNN_INPUT_SIZE = (spectrogram_bands, spectrogram_frames_per_segment)
   

    right_shift_transformation = SpectrogramShift(input_size=CNN_INPUT_SIZE,width_shift_range=4,shift_prob=0.9)
    left_shift_transformation = SpectrogramShift(input_size=CNN_INPUT_SIZE,width_shift_range=4,shift_prob=0.9, left=True)
    random_side_shift_transformation = SpectrogramShift(input_size=CNN_INPUT_SIZE,width_shift_range=4,shift_prob=0.9, random_side=True)

    background_noise_transformation = SpectrogramAddGaussNoise(input_size=CNN_INPUT_SIZE,prob_to_have_noise=0.55)

    dataset = SoundDatasetFold(
                                DATASET_DIR, DATASET_NAME,
                                folds = [1, 2], 
                                shuffle_dataset = True, 
                                generate_spectrograms = True, 
                                shift_transformation = left_shift_transformation,
                                background_noise_transformation = background_noise_transformation,
                                audio_augmentation_pipeline = [],
                                spectrogram_frames_per_segment = spectrogram_frames_per_segment,
                                spectrogram_bands = spectrogram_bands,
                                compute_deltas=True,
                                compute_delta_deltas=True,
                                test = False,
                                progress_bar = True
                            )

    #print("dataset[1]: ",dataset[1])
    
    #dataset = SoundDatasetFold(DATASET_DIR, DATASET_NAME, generate_spectrograms=False,folds=["fold1","fold2","fold3","fold4","fold5","fold6","fold7","fold8", "fold9"])
    #sound, sr = librosa.load(librosa.ex('trumpet'))
    #abs_path = "C:/Users/mikec/Desktop/NN/workspace/UrbanSound8K-CNN-sound-classification/data/UrbanSound8K/audio/fold10/2937-1-0-0.wav"
    #sound, sr = librosa.load(abs_path)
    
    #play_sound(sound,sr)
    #plot_sound_waves(sound, sound_file_name="file.wav", show=True, sound_class="Prova")
    #plot_sound_spectrogram(sound, sound_file_name="file.wav", show=True, sound_class="Prova", log_scale=False)
    #plot_sound_spectrogram(sound, sound_file_name="file.wav", show=True, sound_class="Prova", log_scale=True)
    #plot_sound_spectrogram(sound, sound_file_name="file.wav", show=True, sound_class="Prova", log_scale=True, title="Different hop length", hop_length=2048, sr=22050)
    #plot_periodogram(sound, sound_file_name="file.wav", show=True, sound_class="Prova")
    
    #print(dataset.audio_meta[0])
    #print(dataset.audio_raw[0])
    #sample = dataset[0]
    #print(sample)

    #sample = sample[0]
    #display_heatmap(sample["original_spectrogram"][:,:,0])
    #print(sample["preprocessed_spectrogram"].shape)
    #display_heatmap(sample["preprocessed_spectrogram"][:,:,0])
    #display_heatmap(sample["preprocessed_spectrogram"][:,:,1])
    #display_heatmap(sample["preprocessed_spectrogram"][:,:,2])
    #play_sound(load_audio_file(dataset.data[0]["file_path"])[0])
    #play_sound(dataset.audio_raw[0])

    #progress_bar = tqdm(total=len(dataset), desc="Sample", position=0)
    for i, obj in enumerate(dataset):
        continue
    #    if i>1000: 
    #        break
        #progress_bar.update(1)
    #progress_bar.close()
    #print("mfccs : "+str(sample["mfccs"]))
    #print("chroma: "+str(sample["chroma"]))
    #print("mel: "+ str(sample["mel"]))
    #print("contrast: "+str(sample["contrast"]))
    #print("tonnetz: "+str(sample["tonnetz"]))

    print_code_stats()


    