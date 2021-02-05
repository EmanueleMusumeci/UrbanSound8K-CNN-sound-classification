import os

import random
import math

import copy

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

from core.data_augmentation.image_transformations import *
from core.data_augmentation.audio_transformations import *

from core.utils.dataset_utils import *
from core.utils.audio_utils import *
from core.utils.spectrogram_utils import *
from core.utils.timing import *


'''
The original training time amounted to 40 minutes per epoch (without preprocessing), making training unfeasible
on low-end computers. To improve the training time the dataset was therefore pre-processed (also by pre-generating)
spectrograms) and "compacted" into memory-mapped files using the functions in utils.dataset_utils.

These compacted folds are then loaded when the dataset is instantiated.

To simulate the "randomness" in preprocessing, during the pre-preocessing and compacting phase, given a raw fold,
a preprocessed is generated for each preprocessing value in the preprocessing method (therefore the list of possible)
preprocessing values is discrete, as in the paper https://arxiv.org/pdf/1608.04363v2.pdf). 
Then, at training time, the preprocessing value is extracted randomly and therefore the relative sample is returned.

Please see dataset_utils.py for more.

This solution reduced training time from 40 minutes per epoch (without preprocessing) to just around 3 minutes
per epoch (more than 13X speedup).

NOTICE: When compacting the dataset, a index.json file is generated to keep track of the generated folds and
preprocessing applied to them
'''


class SoundDatasetFold(torch.utils.data.IterableDataset):
    def __init__(self, dataset_dir, dataset_name, 
                folds, 
                audio_meta = None,
                audio_clips = None,
                audio_spectrograms = None,
                
                shuffle = False, 
                
                selected_classes = [1,2,3,4,5,6,7,8,9,10],

                use_spectrograms = True, 

                audio_segment_selector = None,

                image_shift_transformation = None,
                image_background_noise_transformation = None,
                

                audio_clip_duration = 4000,
                sample_rate = 22050,

                spectrogram_bands = 128,
                spectrogram_hop_length = 512,
                normalize_audio = False,
                silent_clip_cutoff_dB = -70,

                preprocessor = None,
                preprocessing_name = None,

                compute_deltas = False,
                compute_delta_deltas = False,
                progress_bar = False,

                test = False,

                debug_preprocessing = False,
                debug_spectrograms = False,

                debug_preprocessing_time = False,

                select_percentage_of_dataset = 1.0,

                load_compacted = True
                ):
        super(SoundDatasetFold).__init__()
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name

        assert len(folds) >= 0, "Provide the list of folds in the loaded dataset"
        if len(folds)==0:
            print("ATTENTION! No folds provided: assuming that the datset will be used in preprocessing mode")
        
        self.folds = folds

        self.selected_classes = selected_classes
        self.num_classes = len(selected_classes)

        self.test_mode = test

        self.audio_meta = audio_meta
        self.audio_clips = audio_clips
        self.audio_spectrograms = audio_spectrograms
        if self.audio_meta is None and folds is not None and len(folds)>0:
            assert self.audio_clips is None and self.audio_spectrograms is None, "If audio_clips or audio_spectrograms were provided, audio_meta should have been provided as well"
            #Load dataset index (the audio files will be loaded one by one therefore training will be up to 10x slower!!!), 
            # returning only the sample_ids whose class_id is among the selected ones
            self.audio_meta, self.sample_ids = self.load_dataset_index(dataset_dir, folds=self.folds)
        
        if self.audio_meta is not None:
            self.sample_ids = self.select_sample_ids(self.selected_classes, select_percentage = select_percentage_of_dataset)        
            class_id_to_name, name_to_class_id, class_id_to_sample_ids = self.index_dataset()
            self.class_id_to_name = class_id_to_name
            self.name_to_class_id = name_to_class_id
            self.class_id_to_sample_ids = class_id_to_sample_ids

            self.class_distribution = self.compute_dataset_distribution()
        else:
            self.sample_ids = []
            self.class_id_to_name = {}
            self.name_to_class_id = {}
            self.class_id_to_sample_ids = {}
            self.class_distribution = {}

        self.preprocessor = preprocessor
        if preprocessor is not None:
            self.preprocessing_name = preprocessor.name
        else:
            self.preprocessing_name = preprocessing_name
            
        if self.preprocessor is not None:
            self.preprocessing_values = self.preprocessor.values
            self.preprocessing_name = self.preprocessor.name
        elif self.preprocessing_name is not None:
            self.preprocessing_values = get_preprocessing_values(dataset_dir, preprocessing_name, folds = folds)
            self.preprocessing_name = preprocessing_name
        else:
            if self.audio_clips is not None:
                assert isinstance(self.audio_clips, list) or isinstance(self.audio_clips, np.ndarray), "If no preprocessing is selected a non-preprocessed dataset should be used"
            elif self.audio_spectrograms is not None:
                assert isinstance(self.audio_spectrograms, list) or isinstance(self.audio_spectrograms, np.ndarray), "If no preprocessing is selected a non-preprocessed dataset should be used"

        self.shuffle = shuffle

        self.use_spectrograms = use_spectrograms

        self.image_shift_transformation = image_shift_transformation
        self.image_background_noise_transformation = image_background_noise_transformation

        self.audio_clip_duration = audio_clip_duration
        self.sample_rate = sample_rate
        
        self.spectrogram_bands = spectrogram_bands

        #If no audio segment selector is specified, a SingleWindowSelector will be constructed to return the whole clip 
        if audio_segment_selector is None:
            self.audio_segment_selector = SingleWindowSelector(4, sample_rate, spectrogram_hop_length=512, random_location = False)
        else:
            assert isinstance(audio_segment_selector, SingleWindowSelector) or isinstance(audio_segment_selector, MultipleWindowSelector), \
                "The audio_segment_selector should be a utils.audio_utils.SingleWindowSelector or utils.audio_utils.MultipleWindowSelector"
            self.audio_segment_selector = audio_segment_selector

        self.silent_clip_cutoff_dB = silent_clip_cutoff_dB

        self.normalize_audio = normalize_audio

        self.compute_deltas = compute_deltas
        self.compute_delta_deltas = compute_delta_deltas
    
        self.progress_bar = progress_bar

        self.debug_spectrograms = debug_spectrograms
        self.debug_preprocessing = debug_preprocessing
        self.debug_preprocessing_time = debug_preprocessing_time

        
    def compute_dataset_distribution(self):
        class_distribution = {}
        for class_id, samples in self.class_id_to_sample_ids.items():
            class_distribution[class_id] = len(samples)
        return class_distribution

    def collect_class_ids(self, skip_first_line=True):
        class_id_to_name = {}
        name_to_class_id = {}

        #run through the whole dataset meta-data csv to detect all class ids and class names
        with open(os.path.join(self.dataset_dir,'UrbanSound8K','metadata','UrbanSound8K.csv'), 'r') as read_obj:
            csv_reader = reader(read_obj)
            
            #next skips the first line that contains the header info
            if skip_first_line: next(csv_reader)

            audios_data_from_csv = []
            for row in csv_reader:
                sample_class_id = int(row[6])
                sample_class_name = row[7]  

                if sample_class_id not in class_id_to_name:
                    class_id_to_name[sample_class_id] = sample_class_name
                    name_to_class_id[sample_class_id] = sample_class_id

        return class_id_to_name, name_to_class_id

    def index_dataset(self):
        
        class_id_to_name, name_to_class_id = self.collect_class_ids()

        class_id_to_sample_ids = {}
        for class_id in self.selected_classes:
            class_id_to_sample_ids[class_id] = []

        for index, sample_meta in enumerate(self.audio_meta):
            sample_meta = self.audio_meta[index]
            sample_class_id = sample_meta["class_id"]
            
            if sample_class_id in self.selected_classes:
                class_id_to_sample_ids[sample_class_id].append(index)

        return class_id_to_name, name_to_class_id, class_id_to_sample_ids
    
    '''
    Selects the "legal" samples based on various criteria:
    1) Select only samples of the selected classes
    2) Select only a certain percentage of the dataset samples (if specified)
    '''
    def select_sample_ids(self, selected_classes, select_percentage = 1.0):
        assert select_percentage<=1 and select_percentage>=0, "Please specify a percentage in range [0,1]"
        sample_ids = []
        select_n_samples = int(len(self.audio_meta) * select_percentage)
        for index, sample_meta in enumerate(self.audio_meta):
            if index > select_n_samples: 
                break
            if int(sample_meta["class_id"]) in selected_classes:
                sample_ids.append(index)
        
        return np.array(sample_ids)

    def get_num_classes(self): return self.num_classes
    def get_id_to_class(self): return self.class_id_to_name

    #The following 3 methods are used in the DataLoader to select which batch fields are going
    #to be collated and tensorized
    def get_preprocessed_fields(self): 
        if self.use_spectrograms:
            return ["preprocessed_spectrogram"]
        else:
            return ["mfccs", "chroma", "mel", "contrast", "tonnetz"]

    def get_unpreprocessed_fields(self): return ["class_name", "meta_data", "original_spectrogram"]
    def get_gold_fields(self): return ["class_id"]

    '''
    Makes this instance callable, returning a preprocessed version on sound, based on how this
    instance was initialized
    '''
    #@function_timer
    def __call__(self, sample_meta, preprocessing_value = None):
        debug = self.debug_preprocessing

        class_id = sample_meta["class_id"]
        class_name = sample_meta["class_name"]
        meta_data = sample_meta["meta_data"]
        
        self.print_debug(sample_meta["file_path"], debug=debug)
    
        try:
            clip, sample_rate = load_audio_file(sample_meta["file_path"], sample_rate=self.sample_rate)
        except:
            raise e

        if self.use_spectrograms:
            original_spectrograms, preprocessed_spectrograms = self.preprocess_convolutional(audio_clip = clip, log_mel_spectrogram=None, \
                                                                                            preprocessor = self.preprocessor, preprocessing_value = preprocessing_value, \
                                                                                            debug=self.debug_preprocessing)
            returned_samples = []
            for orig_spec, prep_spec in zip(original_spectrograms, preprocessed_spectrograms):

                returned_samples.append({
                        "original_spectrogram" : orig_spec, 
                        "preprocessed_spectrogram" : prep_spec, 
                        "class_id" : class_id, 
                        "class_name" : class_name, 
                        "meta_data" : meta_data,
                        "preprocessing_name" : self.preprocessing_name,
                        "preprocessing_value" : preprocessing_value
                        })
            
            self.print_debug(" Returned samples: "+str(len(returned_samples)), debug=debug)
            return returned_samples
        else:
            mfccs, chroma, mel, contrast, tonnetz = self.preprocess_feed_forward(sound, debug=self.debug_preprocessing)
            self.print_debug(" Returned samples: "+str(1), debug=debug)
            return [{
                    "mfccs" : mfccs, 
                    "chroma" : chroma, 
                    "mel" : mel, 
                    "contrast" : contrast, 
                    "tonnetz" : tonnetz, 
                    "class_id" : class_id, 
                    "class_name" : class_name, 
                    "meta_data" : meta_data
                    }]
    
    '''
    Returns the sound clip at index, preprocessed according to how this instance was initialized
    '''
    #@function_timer
    def __getitem__(self, index, debug=False):
        debug = self.debug_preprocessing
        sample = self.audio_meta[index]

        class_id = sample["class_id"]
        class_name = sample["class_name"]
        meta_data = sample["meta_data"]
        
        self.print_debug(sample["file_path"], debug=debug)

        clip = None
        spectrogram = None
        preprocessing_value = None
        #If audio_clips were loaded, we load the correct audio clip at index
        if self.audio_clips is not None:
            #If we are not using a preprocessing, we just select the unpreprocessed audio clip at index
            if (self.preprocessor is None and self.preprocessing_name is None) or self.test_mode:
                clip = self.audio_clips[index]
                preprocessing_value = None
            #else, we first select a preprocessing value and then extract the audio clip from the correct dictionary
            else:
                preprocessing_value = extract_preprocessing_value(self.preprocessing_values)
                clip = self.audio_clips[preprocessing_value][index]

        #else if audio_spectrograms were loaded, we select the spectrogram at index (following a similar pattern to above)
        elif self.audio_spectrograms is not None:
            #as above
            if (self.preprocessor is None and self.preprocessing_name is None) or self.test_mode:
                spectrogram = self.audio_spectrograms[index]
                preprocessing_value = None
            #as above
            else:
                preprocessing_value = extract_preprocessing_value(self.preprocessing_values)
                spectrogram = self.audio_spectrograms[preprocessing_value][index]
        else:
            try:
                clip, sample_rate = load_audio_file(sample["file_path"], sample_rate=self.sample_rate)
            except:
                raise e

        if self.use_spectrograms:
            original_spectrograms, preprocessed_spectrograms = self.preprocess_convolutional(audio_clip = clip, log_mel_spectrogram=spectrogram, \
                                                                                            preprocessor = self.preprocessor, preprocessing_value = preprocessing_value, \
                                                                                            debug=self.debug_preprocessing)
            returned_samples = []
            for orig_spec, prep_spec in zip(original_spectrograms, preprocessed_spectrograms):

                returned_samples.append({
                        "original_spectrogram" : orig_spec, 
                        "preprocessed_spectrogram" : prep_spec, 
                        "class_id" : class_id, 
                        "class_name" : class_name, 
                        "meta_data" : meta_data,
                        "preprocessing_name" : self.preprocessing_name,
                        "preprocessing_value" : preprocessing_value
                        })
            
            self.print_debug(" Returned samples: "+str(len(returned_samples)), debug=debug)
            return returned_samples
        else:
            mfccs, chroma, mel, contrast, tonnetz = self.preprocess_feed_forward(sound, debug=self.debug_preprocessing)
            self.print_debug(" Returned samples: "+str(1), debug=debug)
            return [{
                    "mfccs" : mfccs, 
                    "chroma" : chroma, 
                    "mel" : mel, 
                    "contrast" : contrast, 
                    "tonnetz" : tonnetz, 
                    "class_id" : class_id, 
                    "class_name" : class_name, 
                    "meta_data" : meta_data
                    }]
        
    '''
    Makes this instance iterable, iterating through all the sample whose id is considered in this dataset
    (see select_sample_ids) and preprocessing these samples according to how this instance was initialized
    '''
    def __iter__(self):
        if self.shuffle: self.shuffle_dataset()

        if self.progress_bar:
            progress_bar = tqdm(total=len(self.sample_ids), desc="Sample", position=0)
        for index in self.sample_ids:
            preprocessed_samples = self[index]

            #In case the sample was dropped because too silent
            if len(preprocessed_samples) == 0:
                if self.progress_bar:
                    progress_bar.update(1)
                continue

            previous_samples = preprocessed_samples

            if self.progress_bar:
                progress_bar.update(1)
            for sample in preprocessed_samples:
                yield sample

        if self.progress_bar:
            progress_bar.close()
        
    '''
    NOT TESTED - NOT USED
    Applies the preprocessing required for the feed forward network
    '''
    def preprocess_feed_forward(self, audio_clip, debug=False):
    
        sample_rate = self.sample_rate

        if self.normalize_audio:
            audio_clip = normalize_clip(audio_clip)

        self.print_debug("Features :"+str(len(audio_clip))+"sampled at "+str(sample_rate)+"hz", debug=debug)
        #Short-time Fourier transform(STFT)

        stft = np.abs(librosa.stft(audio_clip))
        self.print_debug("stft:\n"+str(stft), debug=debug)
        #Mel-frequency cepstral coefficients (MFCCs)
        self.print_debug("before mfccs:\n"+str(stft), debug=debug)
        mfccs = np.mean(librosa.feature.mfcc(S=audio_clip, sr=sample_rate, n_mfcc=40).T,axis=0)
        mfccs = mfccs[..., np.newaxis]
        self.print_debug("mfccs:\n"+str(mfccs), debug=debug)

        #Compute a chromagram from a waveform or power spectrogram.
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        self.print_debug("chroma:\n"+str(chroma), debug=debug)

        #Compute a mel-scaled spectrogram.
        #SLOWER mel = np.mean(librosa.feature.melspectrogram(audio_clip, sr=sample_rate).T,axis=0)
        
        #https://librosa.org/doc/main/generated/librosa.feature.melspectrogram.html
        #avoid repeating stft computation in mel_spectrogram
        power_spectrogram = np.abs(stft)**2
        mel = np.mean(librosa.feature.melspectrogram(S=power_spectrogram, sr=sample_rate).T,axis=0)
        self.print_debug("mel:\n"+str(mel), debug=debug)

        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
        self.print_debug("contrast:\n"+str(contrast), debug=debug)

        #The warning is triggered by this problem: https://github.com/librosa/librosa/issues/1214
        #Computes the tonal centroid features (tonnetz)
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(audio_clip), sr=sample_rate).T,axis=0)
        self.print_debug("tonnetz:\n"+str(tonnetz), debug=debug)
        
        return mfccs, chroma, mel, contrast, tonnetz
    
    '''
    Applies the preprocessing required for the convolutional networks
    '''
    #@function_timer
    def preprocess_convolutional(self, audio_clip = None, log_mel_spectrogram = None, preprocessor = None, preprocessing_value = None, debug = False):
        
        assert audio_clip is not None or log_mel_spectrogram is not None,\
             "preprocess_convolutional should receive at least one among audio_clip and original_log_mel_spectrogram"

        audio_clip_length = None
        spectrogram_length = None
        if audio_clip is not None:
            if self.normalize_audio:
                self.print_debug("Normalizing audio clip", debug=debug)
                preprocessed_audio_clip = normalize_clip(audio_clip)
            
            if preprocessor is not None and not self.test_mode:
                self.print_debug("Preprocessing audio clip", debug=debug)
                preprocessed_audio_clip = preprocessor(audio_clip, value = preprocessing_value)
            else:
                preprocessed_audio_clip = audio_clip
            
            audio_clip_length = len(preprocessed_audio_clip)
        elif log_mel_spectrogram is not None:
            spectrogram_length = log_mel_spectrogram.shape[1]
      
        original_spectrograms = []
        preprocessed_spectrograms = []
        
        self.print_debug("Segments: "+str(list(self.audio_segment_selector(audio_clip_length, spectrogram_length))), debug=debug)
        
        for i, segment_bounds in enumerate(self.audio_segment_selector(audio_clip_length, spectrogram_length)):
            
            if debug: 
                self.print_debug("Preprocessing segment: "+str(segment_bounds), debug=debug)

            #if audio_clip is None, segment_start and segment_end will be None
            #if log_mel_spectrogram is None, spectrogram_start, spectrogram_end will be None
            segment_start, segment_end, spectrogram_start, spectrogram_end = segment_bounds
            
            #Only accept audio clip segments that are:
            #1) Of a fixed window size (self.spectrogram_window_size)
            #2) Fully contained in the audio clip (segments that go "out of bounds" wrt the audio clip are not considered)
            if debug: 
                if audio_clip is not None:
                    self.print_debug("Audio segment: ("+str(segment_start)+", "+str(segment_end)+") of length ("+str(segment_end - segment_start)+")", debug=debug)
                if log_mel_spectrogram is not None:
                    self.print_debug("Spectrogram segment: ("+str(spectrogram_start)+", "+str(spectrogram_end)+") of length ("+str(spectrogram_end - spectrogram_start)+")", debug=debug)
            
            if log_mel_spectrogram is not None:
                log_mel_spectrogram_segment = log_mel_spectrogram[:,spectrogram_start:spectrogram_end]
            else:
                original_signal_segment = audio_clip[segment_start:segment_end]
                log_mel_spectrogram_segment = generate_mel_spectrogram_librosa(original_signal_segment, self.spectrogram_bands, debug_time_label=("original" if self.debug_preprocessing_time else ""), show=self.debug_spectrograms)
                preprocessed_signal_segment = preprocessed_audio_clip[segment_start:segment_end]
                
            
            with code_timer("drop silent", debug=self.debug_preprocessing_time):
                if self.silent_clip_cutoff_dB is not None:
                    #drop silent frames (taken from https://github.com/karolpiczak/paper-2015-esc-convnet/blob/master/Code/_Datasets/Setup.ipynb)
                    #ONLY if they aren't the only frame of the audio clip
                    mean_value = np.mean(log_mel_spectrogram_segment)
                    self.print_debug("Mean dB value:" +str(mean_value), debug=debug)
                    if mean_value <= self.silent_clip_cutoff_dB:
                        self.print_debug("Silent segment dropped", debug=debug)
                        continue
                
            if log_mel_spectrogram is not None:
                preprocessed_log_mel_spectrogram_segment = log_mel_spectrogram_segment
            else:
                self.print_debug("Generating mel spectrogram", debug=debug)
                preprocessed_log_mel_spectrogram_segment = generate_mel_spectrogram_librosa(preprocessed_signal_segment, self.spectrogram_bands, debug_time_label=("preprocessed" if self.debug_preprocessing_time else ""), show=self.debug_spectrograms)

            original_spectrograms.append(log_mel_spectrogram_segment)
            preprocessed_spectrograms.append(preprocessed_log_mel_spectrogram_segment)

        if log_mel_spectrogram is not None:
            with code_timer("original spectrogram reshape", debug=self.debug_preprocessing_time):
            #Reshape the spectrograms from [N_AUDIO_SEGMENT, N_BANDS, N_FRAMES] to [N_AUDIO_SEGMENT, N_BANDS, N_FRAMES, 1]
                original_spectrograms = np.asarray(original_spectrograms).reshape(len(original_spectrograms),self.spectrogram_bands,self.audio_segment_selector.spectrogram_window_size,1)

        with code_timer("preprocessed spectrogram reshape", debug=self.debug_preprocessing_time):
            preprocessed_spectrograms = np.asarray(preprocessed_spectrograms).reshape(len(preprocessed_spectrograms),self.spectrogram_bands,self.audio_segment_selector.spectrogram_window_size,1)

        with code_timer("deltas", debug=self.debug_preprocessing_time):
            preprocessed_spectrograms_shape = np.shape(preprocessed_spectrograms)
            if self.compute_deltas:
                self.print_debug("Computing spectrogram delta", debug=debug)
                #Make space for the delta features
                preprocessed_spectrograms = np.concatenate((preprocessed_spectrograms, np.zeros(preprocessed_spectrograms_shape)), axis = 3)            

                if self.compute_delta_deltas:
                    self.print_debug("Computing spectrogram delta-delta", debug=debug)
                    preprocessed_spectrograms = np.concatenate((preprocessed_spectrograms, np.zeros(preprocessed_spectrograms_shape)), axis = 3)

                for i in range(len(preprocessed_spectrograms)):
                    preprocessed_spectrograms[i, :, :, 1] = librosa.feature.delta(preprocessed_spectrograms[i, :, :, 0])
                    
                    if self.compute_delta_deltas:
                        preprocessed_spectrograms[i, :, :, 2] = librosa.feature.delta(preprocessed_spectrograms[i, :, :, 1])

            #preprocessed_spectrograms is the preprocessed output with shape [N_AUDIO_SEGMENTS, N_BANDS, N_FRAMES, N_FEATURES] where
            #N_FEATURES is 1, 2 or 3 depending on our choice of computing delta and delta-delta spectrograms
        

        with code_timer("image augmentation", debug=self.debug_preprocessing_time):
            if not self.test_mode:
                #Spectrogram image transformation
                for i in range(len(preprocessed_spectrograms)):
                    #Background noise transformations should not be applied to deltas
                    if self.image_background_noise_transformation is not None:
                        self.print_debug("Applying background noise image augmentation to spectrogram", debug=debug)
                        preprocessed_spectrogram, noise_mask = self.image_background_noise_transformation(preprocessed_spectrograms[i, :, :, 0])
                        preprocessed_spectrograms[i, :, :, 0] = preprocessed_spectrogram

                    if self.image_shift_transformation is not None:
                        self.print_debug("Applying right shift image augmentation to spectrogram", debug=debug)
                        shifted_spectrogram, shift_position = self.image_shift_transformation(preprocessed_spectrograms[i, :, :, 0])
                        preprocessed_spectrograms[i, :, :, 0] = shifted_spectrogram

                        #Spectrogram shift transformations can be applied to delta and delta-delta histograms as well
                        if self.compute_deltas:
                            self.print_debug("Applying right shift image augmentation to spectrogram delta", debug=debug)
                            shifted_delta_spectrogram, _ = self.image_shift_transformation(preprocessed_spectrograms[i, :, :, 1], shift_position=shift_position)
                            preprocessed_spectrograms[i, :, :, 1] = shifted_delta_spectrogram
                            if self.compute_delta_deltas:
                                self.print_debug("Applying right shift image augmentation to spectrogram delta-delta", debug=debug)
                                shifted_delta_delta_spectrogram, _ = self.image_shift_transformation(preprocessed_spectrograms[i, :, :, 2], shift_position=shift_position)
                                preprocessed_spectrograms[i, :, :, 2] = shifted_delta_delta_spectrogram

        if log_mel_spectrogram is not None:
            self.print_debug("Original spectrograms shape: "+str(original_spectrograms.shape), debug=debug)
        
        self.print_debug("Preprocessed spectrograms shape: "+str(preprocessed_spectrograms.shape), debug=debug)

        return original_spectrograms, preprocessed_spectrograms
    
    '''
    Decode integer class ids to strings
    '''
    def decode_class_names(self, class_indices):
        return [self.class_id_to_name[idx] for idx in class_indices]

    '''
    Shuffle the dataset. To avoid shuffling the whole dataset, which might be using memory-mapped files,
    a index list is shuffled instead and then the samples are iterated through or extracted by index
    according to the order of indices in this list
    '''
    def shuffle_dataset(self):
        p = np.random.permutation(len(self.sample_ids))
        self.sample_ids = self.sample_ids[p.astype(int)]

    '''
    Loads the UrbanSound8K dataset, storing its meta-data (including the sound clip file path) and its indices
    (that will be later used for shuffling)
    '''
    def load_dataset_index(self, sample, folds = [], skip_first_line=True):
        with open(os.path.join(self.dataset_dir,'UrbanSound8K','metadata','UrbanSound8K.csv'), 'r') as read_obj:
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
                fold_number = int(audio[5])
                if fold_number in self.folds:
                    
                    metadata = {
                        "fsID":audio[1],
                        "start":audio[2],
                        "end":audio[3],
                        "salience":audio[4]
                    }   
                    audiodict = {
                        "file_path": os.path.join(self.dataset_dir, self.dataset_name, "audio", "fold"+str(fold_number), audio[0]),
                        "class_id":int(audio[6]),
                        "class_name":audio[7],
                        "meta_data": metadata
                    }

                    audio_meta.append(audiodict)
                    audio_ids.append(index)
                    index += 1
        return audio_meta, audio_ids     

    '''
    Prints to stdout preserving the tqdm progress bar position
    '''
    def print_debug(self, string, debug, **kwargs):
        if not debug: return
        if self.progress_bar:
            tqdm.write(str(string))
        else:
            print(string, **kwargs)