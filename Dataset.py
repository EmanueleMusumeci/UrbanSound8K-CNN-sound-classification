import os

import random
import math

import copy

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

try:
    from data_augmentation.image_transformations import *
    from data_augmentation.audio_transformations import *

    from utils.dataset_utils import *
    from utils.audio_utils import *
    from utils.spectrogram_utils import *
    from utils.timing import *
except:
    pass


class SoundDatasetFold(torch.utils.data.IterableDataset):
    def __init__(self, dataset_dir, dataset_name, 
                folds, 
                audio_meta = None,
                audio_clips = None,
                audio_spectrograms = None,
                
                shuffle = False, 
                
                selected_classes = [0,1,2,3,4,5,6,7,8,9],

                use_spectrograms = True, 

                audio_segment_selector = None,

                image_shift_transformation = None,
                image_background_noise_transformation = None,
                
                #time_stretch_transformation = None,
                #pitch_shift_transformation = None,

                audio_clip_duration = 4000,
                sample_rate = 22050,

                spectrogram_bands = 60,
                spectrogram_hop_length = 512,
                spectrogram_frames_per_segment = 41,
                #spectrogram_window_overlap = 0.5,
                #drop_last_spectrogram_frame = True,
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

        assert len(folds) > 0, "Provide the list of folds in the loaded dataset"
        self.folds = folds

        self.selected_classes = selected_classes
        self.num_classes = len(selected_classes)

        self.test_mode = test

        self.audio_meta = audio_meta
        self.audio_clips = audio_clips
        self.audio_spectrograms = audio_spectrograms

        if self.audio_meta is None:
            assert self.audio_clips is None and self.audio_spectrograms is None, "If audio_clips or audio_spectrograms were provided, audio_meta should have been provided as well"
            #Load dataset index (the audio files will be loaded one by one therefore training will be up to 10x slower!!!), 
            # returning only the sample_ids whose class_id is among the selected ones
            self.audio_meta = self.load_dataset_index(dataset_dir, folds=self.folds)

        self.preprocessor = preprocessor
        self.preprocessing_name = preprocessing_name
        if self.preprocessor is not None:
            raise
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

        self.sample_ids = self.select_sample_ids(self.selected_classes, select_percentage = select_percentage_of_dataset)        

        self.shuffle = shuffle

        self.use_spectrograms = use_spectrograms

        self.image_shift_transformation = image_shift_transformation
        self.image_background_noise_transformation = image_background_noise_transformation

        #self.time_stretch_transformation = time_stretch_transformation
        #self.pitch_shift_transformation = pitch_shift_transformation

        self.audio_clip_duration = audio_clip_duration
        self.sample_rate = sample_rate
        
        #as in https://github.com/karolpiczak/paper-2015-esc-convnet/blob/master/Code/_Datasets/Setup.ipynb
        #self.spectrogram_frames_per_segment = spectrogram_frames_per_segment
        #self.spectrogram_hop_length = spectrogram_hop_length
        #self.spectrogram_window_size = self.spectrogram_hop_length * (spectrogram_frames_per_segment-1)
        #assert spectrogram_window_overlap > 0 and spectrogram_window_overlap < 1, "spectrogram_window_overlap should be between 0 and 1"
        #self.spectrogram_window_overlap = spectrogram_window_overlap
        #self.spectrogram_window_step_size = math.floor(self.spectrogram_window_size * (1-spectrogram_window_overlap))
        self.spectrogram_bands = spectrogram_bands
        #self.drop_last_spectrogram_frame = drop_last_spectrogram_frame

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

        class_id_to_name, name_to_class_id, class_id_to_sample_ids = self.index_dataset()
        self.class_id_to_name = class_id_to_name
        self.name_to_class_id = name_to_class_id
        self.class_id_to_sample_ids = class_id_to_sample_ids

        self.class_distribution = self.compute_dataset_distribution()
        
# TODO MICHELE
    # Per ogni classe , numero di istanze 
    def compute_dataset_distribution(self):
        class_distribution = {}
        for class_id, samples in self.class_id_to_sample_ids.items():
            class_distribution[class_id] = len(samples)
        return class_distribution

    def index_dataset(self):
        class_id_to_name = {}
        name_to_class_id = {}
        
        class_id_to_sample_ids = {}
        for class_id in self.selected_classes:
            class_id_to_sample_ids[class_id] = []

        for index, sample_meta in enumerate(self.audio_meta):
            sample_meta = self.audio_meta[index]
            sample_class_id = int(sample_meta["class_id"])
            sample_class_name = sample_meta["class_name"]

            if sample_class_id not in class_id_to_name:
                class_id_to_name[sample_class_id] = sample_class_name
                name_to_class_id[sample_class_id] = sample_class_id
            
            if sample_class_id in self.selected_classes:
                class_id_to_sample_ids[sample_class_id].append(index)

        return class_id_to_name, name_to_class_id, class_id_to_sample_ids
    
    '''
    Selects the "legal" samples based on various criteria:
    1) Select only samples of the selecte classes
    2) Select only a certain percentage of the datase samples (if specified)
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

    def get_preprocessed_fields(self): 
        if self.use_spectrograms:
            return ["preprocessed_spectrogram"]
        else:
            return ["mfccs", "chroma", "mel", "contrast", "tonnetz"]

    def get_unpreprocessed_fields(self): return ["class_name", "meta_data", "original_spectrogram"]
    def get_gold_fields(self): return ["class_id"]

    def __call__(self, sound, sample_rate=22050):
        if self.use_spectrograms:
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

        clip = None
        spectrogram = None
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
            
            if debug: print(" Returned samples: "+str(len(returned_samples)))
            return returned_samples
        else:
            mfccs, chroma, mel, contrast, tonnetz = self.preprocess_feed_forward(sound, debug=self.debug_preprocessing)
            if debug: print(" Returned samples: "+str(1))
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
    
    #pattern iter, chiamo n=len(dataset) volte __getitem__ (nostro metodo getter)
    #con yield
    #forse spacchetto dati di __getitem__, e.q se due finestre, una alla volta
    def __iter__(self):
        if self.shuffle: self.shuffle_dataset()

        if self.progress_bar:
            progress_bar = tqdm(total=len(self.sample_ids), desc="Sample", position=0)
        for index in self.sample_ids:
            #try:
            preprocessed_samples = self[index]
            previous_samples = preprocessed_samples
            #This is a temporary fix to a decoding error with files from clip 36429: we provide the last sample twice 
            #except pydub.exceptions.CouldntDecodeError:
            #    preprocessed_samples = previous_samples

            if self.progress_bar:
                progress_bar.update(1)
            for sample in preprocessed_samples:
                yield sample

        if self.progress_bar:
            progress_bar.close()

    def preprocess_feed_forward(self, audio_clip, debug=False):
    
        sample_rate = self.sample_rate

        if self.normalize_audio:
            audio_clip = normalize_clip(audio_clip)

        if debug: print("Features :"+str(len(audio_clip))+"sampled at "+str(sample_rate)+"hz")
        #Short-time Fourier transform(STFT)

        stft = np.abs(librosa.stft(audio_clip))
        if debug: print("stft:\n"+str(stft))
        #Mel-frequency cepstral coefficients (MFCCs)
        if debug: print("before mfccs:\n"+str(stft))
        mfccs = np.mean(librosa.feature.mfcc(S=audio_clip, sr=sample_rate, n_mfcc=40).T,axis=0)
        mfccs = mfccs[..., np.newaxis]
        if debug: print("mfccs:\n"+str(mfccs))

        #Compute a chromagram from a waveform or power spectrogram.
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        if debug: print("chroma:\n"+str(chroma))

        #Compute a mel-scaled spectrogram.
        #SLOWER mel = np.mean(librosa.feature.melspectrogram(audio_clip, sr=sample_rate).T,axis=0)
        
        #https://librosa.org/doc/main/generated/librosa.feature.melspectrogram.html
        #avoid repeating stft computation in mel_spectrogram
        power_spectrogram = np.abs(stft)**2
        mel = np.mean(librosa.feature.melspectrogram(S=power_spectrogram, sr=sample_rate).T,axis=0)
        if debug: print("mel:\n"+str(mel))

        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
        if debug: print("contrast:\n"+str(contrast))

        #The warning is triggered by this problem: https://github.com/librosa/librosa/issues/1214
        #Computes the tonal centroid features (tonnetz)
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(audio_clip), sr=sample_rate).T,axis=0)
        if debug: print("tonnetz:\n"+str(tonnetz))
        
        return mfccs, chroma, mel, contrast, tonnetz

#TODO add preprocessed_audio_clip and preprocessed_log_mel_spectrogram as arguments and change code accordingly
    #@function_timer
    def preprocess_convolutional(self, audio_clip = None, log_mel_spectrogram = None, preprocessor = None, preprocessing_value = None, debug = False):
        
        assert audio_clip is not None or log_mel_spectrogram is not None,\
             "preprocess_convolutional should receive at least one among audio_clip and original_log_mel_spectrogram"

        audio_clip_length = None
        spectrogram_length = None
        if audio_clip is not None:
            if self.normalize_audio:
                preprocessed_audio_clip = normalize_clip(audio_clip)
            
            if preprocessor is not None and not self.test_mode:
                preprocessed_audio_clip = preprocessor(audio_clip, value = preprocessing_value)
            
            audio_clip_length = len(preprocessed_audio_clip)
        elif log_mel_spectrogram is not None:
            spectrogram_length = log_mel_spectrogram.shape[1]
        
        #TO DELETE
        #if self.time_stretch_transformation is not None:
        #    if not self.test_mode:
        #        with code_timer("audio time stretch", debug=self.debug_preprocessing_time):
        #            audio_clip = self.time_stretch_transformation(audio_clip)
        
        original_spectrograms = []
        preprocessed_spectrograms = []
        
        if debug: print("segments: "+str(list(self.audio_segment_selector(audio_clip_length, spectrogram_length))))
        
        for i, segment_bounds in enumerate(self.audio_segment_selector(audio_clip_length, spectrogram_length)):
            
            if debug: 
                print(segment_bounds)

            #if audio_clip is None, segment_start and segment_end will be None
            #if log_mel_spectrogram is None, spectrogram_start, spectrogram_end will be None
            segment_start, segment_end, spectrogram_start, spectrogram_end = segment_bounds
            
            #Only accept audio clip segments that are:
            #1) Of a fixed window size (self.spectrogram_window_size)
            #2) Fully contained in the audio clip (segments that go "out of bounds" wrt the audio clip are not considered)
            #if len(original_signal_segment) == math.floor(self.spectrogram_window_size) and segment_end<len(audio_clip): 
            if debug: 
                print("audio segment ("+str(segment_start)+", "+str(segment_end)+")")
                if log_mel_spectrogram is not None:
                    print("spectrogram segment ("+str(spectrogram_start)+", "+str(spectrogram_end)+")")
            
            if log_mel_spectrogram is not None:
                log_mel_spectrogram_segment = log_mel_spectrogram[:,spectrogram_start:spectrogram_end]
            else:
                original_signal_segment = audio_clip[segment_start:segment_end]
                preprocessed_signal_segment = preprocessed_audio_clip[segment_start:segment_end]
                
            #display_heatmap(original_mel_spectrogram_librosa)
            #raise
            
            with code_timer("drop silent", debug=self.debug_preprocessing_time):
                if self.silent_clip_cutoff_dB is not None:
                    #drop silent frames (taken from https://github.com/karolpiczak/paper-2015-esc-convnet/blob/master/Code/_Datasets/Setup.ipynb)
                    #ONLY if they aren't the only frame of the audio clip
                    if debug: print("Mean dB value:" +str(np.mean(log_mel_spectrogram_segment)))
                    if i>0 and np.mean(log_mel_spectrogram_segment) <= self.silent_clip_cutoff_dB:
                        if debug: print("Silent segment dropped")
                        continue
                
            if log_mel_spectrogram is not None:
                preprocessed_log_mel_spectrogram_segment = log_mel_spectrogram_segment
            else:
                log_mel_spectrogram_segment = generate_mel_spectrogram_librosa(original_signal_segment, self.spectrogram_bands, debug_time_label=("original" if self.debug_preprocessing_time else ""), show=self.debug_spectrograms)
                preprocessed_log_mel_spectrogram_segment = generate_mel_spectrogram_librosa(preprocessed_signal_segment, self.spectrogram_bands, debug_time_label=("preprocessed" if self.debug_preprocessing_time else ""), show=self.debug_spectrograms)

            original_spectrograms.append(log_mel_spectrogram)
            preprocessed_spectrograms.append(preprocessed_log_mel_spectrogram_segment)

            #TO DELETE
            #if self.pitch_shift_transformation is not None:
            #    if not self.test_mode:
            #        with code_timer("audio pitch shift", debug=self.debug_preprocessing_time):
            #            preprocessed_signal_segment = self.pitch_shift_transformation(preprocessed_signal_segment)
        
        if log_mel_spectrogram is None:
            with code_timer("original spectrogram reshape", debug=self.debug_preprocessing_time):
            #Reshape the spectrograms from [N_AUDIO_SEGMENT, N_BANDS, N_FRAMES] to [N_AUDIO_SEGMENT, N_BANDS, N_FRAMES, 1]
                original_spectrograms = np.asarray(original_spectrograms).reshape(len(original_spectrograms),self.spectrogram_bands,self.audio_segment_selector.spectrogram_window_size,1)
        
        with code_timer("preprocessed spectrogram reshape", debug=self.debug_preprocessing_time):
            preprocessed_spectrograms = np.asarray(preprocessed_spectrograms).reshape(len(preprocessed_spectrograms),self.spectrogram_bands,self.audio_segment_selector.spectrogram_window_size,1)
        
        with code_timer("deltas", debug=self.debug_preprocessing_time):
          if self.compute_deltas:
                #Make space for the delta features
                preprocessed_spectrograms = np.concatenate((preprocessed_spectrograms, np.zeros(np.shape(preprocessed_spectrograms))), axis = 3)
                if self.compute_delta_deltas:
                    preprocessed_spectrograms = np.concatenate((preprocessed_spectrograms, np.zeros(np.shape(preprocessed_spectrograms))), axis = 3)

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
                        preprocessed_spectrogram, noise_mask = self.image_background_noise_transformation(preprocessed_spectrograms[i, :, :, 0])
                        preprocessed_spectrograms[i, :, :, 0] = preprocessed_spectrogram

                    if self.image_shift_transformation is not None:
                        shifted_spectrogram, shift_position = self.image_shift_transformation(preprocessed_spectrograms[i, :, :, 0])
                        preprocessed_spectrograms[i, :, :, 0] = shifted_spectrogram

                        #Spectrogram shift transformations can be applied to delta and delta-delta histograms as well
                        if self.compute_deltas:
                            shifted_delta_spectrogram, _ = self.image_shift_transformation(preprocessed_spectrograms[i, :, :, 1], shift_position=shift_position)
                            preprocessed_spectrograms[i, :, :, 1] = shifted_delta_spectrogram
                            if self.compute_delta_deltas:
                                shifted_delta_delta_spectrogram, _ = self.image_shift_transformation(preprocessed_spectrograms[i, :, :, 2], shift_position=shift_position)
                                preprocessed_spectrograms[i, :, :, 2] = shifted_delta_delta_spectrogram
        

        return original_spectrograms, preprocessed_spectrograms

    def shuffle_dataset(self):
        p = np.random.permutation(len(self.sample_ids))
        self.sample_ids = self.sample_ids[p.astype(int)]

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
                        "class_id":int(audio[6]),
                        "class_name":audio[7],
                        "meta_data": metadata
                    }

                    audio_meta.append(audiodict)
                    audio_ids.append(index)
                    index += 1
        return audio_meta, audio_ids     
            
    #def __len__(self):
    #    #if self.use_spectrograms:
    #    #    return len(self.data) * self.fixed_segment_size
    #    #else:
    #    return len(self.data)

if __name__ == "__main__":
    import os

    import torch
    try:
        from Dataset import SoundDatasetFold
        from DataLoader import DataLoader
        from nn.convolutional_model import ConvolutionalNetwork
        from nn.feed_forward_model import FeedForwardNetwork
        from data_augmentation.image_transformations import *
        from data_augmentation.audio_transformations import *
        from Trainer import *
        from utils.audio_utils import SingleWindowSelector, MultipleWindowSelector
    except:
        pass

    INSTANCE_NAME = "PROVA"
    BATCH_SIZE = 128
    USE_CNN = True
    APPLY_IMAGE_AUGMENTATIONS = False
    APPLY_AUDIO_AUGMENTATIONS = False

    DEBUG_PREPROCESSING = True
    DEBUG_TIMING = True

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        from google.colab import drive
        base_dir = "/content/drive/My Drive/Neural Networks Project"
        DATASET_DIR = "/content/drive/data"
    except:
        base_dir = os.path.dirname(os.path.realpath(__file__))
        DATASET_DIR = os.path.join(base_dir,"data")

    DATASET_NAME = "UrbanSound8K"
    DATASET_PERCENTAGE = 1.0

    MODEL_DIR = os.path.join(base_dir,"model")

    selected_classes = [0,1,2,3,4,5,6,7,8,9]

    spectrogram_frames_per_segment = 128
    spectrogram_bands = 128
    in_channels = (3 if USE_CNN else 1)

    CNN_INPUT_SIZE = (spectrogram_bands, spectrogram_frames_per_segment, in_channels)
    FFN_INPUT_SIZE = 154

    #Image augmentations
    if APPLY_IMAGE_AUGMENTATIONS:
        right_shift_transformation = SpectrogramShift(input_size=CNN_INPUT_SIZE,width_shift_range=4,shift_prob=0.9)
        left_shift_transformation = SpectrogramShift(input_size=CNN_INPUT_SIZE,width_shift_range=4,shift_prob=0.9, left=True)
        random_side_shift_transformation = SpectrogramShift(input_size=CNN_INPUT_SIZE,width_shift_range=4,shift_prob=0.9, random_side=True)
        background_noise_transformation = SpectrogramAddGaussNoise(input_size=CNN_INPUT_SIZE,prob_to_have_noise=0.55)
    else:
        right_shift_transformation = None
        left_shift_transformation = None
        random_side_shift_transformation = None
        background_noise_transformation = None

    #Audio augmentations
    #if APPLY_AUDIO_AUGMENTATIONS:
    #    random_pitch_shift = PitchShift([-3.5, -2.5, 2.5, 3.5], debug_time=DEBUG_TIMING)
    #    random_time_stretch = TimeStretch([0.81, 0.93, 1.07, 1.23], debug_time=DEBUG_TIMING)
    #else:
    #    random_pitch_shift = None
    #    random_time_stretch = None

    preprocessing_name = None
    preprocessing_name = "PitchShift"

    train_fold_list = [1]
    train_fold_list = [1,2,3,4,5,6,7,8,9]
    test_fold_list = [10]

    if preprocessing_name is not None:
        train_audio_meta, train_audio_clips, train_audio_spectrograms = load_preprocessed_compacted_dataset(DATASET_DIR, preprocessing_name, folds = train_fold_list)
        #_, _, raw_train_audio_spectrograms = load_raw_compacted_dataset(DATASET_DIR, folds = train_fold_list)
    else:
        train_audio_meta, train_audio_clips, train_audio_spectrograms = load_raw_compacted_dataset(DATASET_DIR, folds = train_fold_list)

    test_audio_meta, test_audio_clips, test_audio_spectrograms = load_raw_compacted_dataset(DATASET_DIR, folds = test_fold_list)

    train_dataset = SoundDatasetFold(DATASET_DIR, DATASET_NAME, 
                                folds = train_fold_list, 
                                preprocessing_name = preprocessing_name,
                                audio_meta = train_audio_meta,
                                audio_clips = None,
                                audio_spectrograms = train_audio_spectrograms,
                                shuffle = True, 
                                use_spectrograms = USE_CNN, 
                                image_shift_transformation = right_shift_transformation, 
                                image_background_noise_transformation = background_noise_transformation, 
                                #time_stretch_transformation = random_time_stretch,
                                #pitch_shift_transformation = random_pitch_shift, 
                                spectrogram_frames_per_segment = spectrogram_frames_per_segment, 
                                spectrogram_bands = spectrogram_bands, 
                                compute_deltas=False, 
                                compute_delta_deltas=False, 
                                test = False, 
                                progress_bar = False,
                                selected_classes=selected_classes,
                                select_percentage_of_dataset=DATASET_PERCENTAGE,
                                audio_segment_selector=SingleWindowSelector(3, spectrogram_hop_length=512, random_location = True),
                                debug_preprocessing=DEBUG_PREPROCESSING,
                                debug_preprocessing_time=DEBUG_TIMING,
                                silent_clip_cutoff_dB = None
                                )   

    test_dataset = SoundDatasetFold(DATASET_DIR, DATASET_NAME,  
                                folds = test_fold_list, 
                                preprocessing_name = preprocessing_name,
                                audio_meta = test_audio_meta,
                                audio_clips = None,
                                audio_spectrograms = test_audio_spectrograms,
                                shuffle = False, 
                                use_spectrograms = USE_CNN, 
                                spectrogram_frames_per_segment = spectrogram_frames_per_segment, 
                                spectrogram_bands = spectrogram_bands, 
                                compute_deltas=False, 
                                compute_delta_deltas=False, 
                                test = True, 
                                progress_bar = True,
                                selected_classes=selected_classes,
                                select_percentage_of_dataset=DATASET_PERCENTAGE,
                                audio_segment_selector=SingleWindowSelector(3, spectrogram_hop_length=512),
                                debug_preprocessing=DEBUG_PREPROCESSING,
                                debug_preprocessing_time=DEBUG_TIMING,
                                silent_clip_cutoff_dB = None
                                )

    '''
    display_heatmap(raw_train_audio_spectrograms[train_dataset.sample_ids[0]])
    train_sample = train_dataset[0]
    display_heatmap(train_sample[0]["original_spectrogram"])
    print(train_sample[0]["original_spectrogram"].shape)
    display_heatmap(train_sample[0]["preprocessed_spectrogram"][:,:,0])
    print(train_sample[0]["preprocessed_spectrogram"][:,:,0].shape)
    

    display_heatmap(test_audio_spectrograms[test_dataset.sample_ids[0]])
    test_sample = test_dataset[0]
    display_heatmap(test_sample[0]["original_spectrogram"])
    print(test_sample[0]["original_spectrogram"].shape)
    display_heatmap(test_sample[0]["preprocessed_spectrogram"][:,:,0])
    print(test_sample[0]["preprocessed_spectrogram"][:,:,0].shape)
    '''

    #test_sample = test_dataset[0]
    #print(test_sample["original_spectrogram"])
    #print(test_sample["preprocessed_spectrogram"])
    #progress_bar = tqdm(total=len(dataset), desc="Sample", position=0)
    with code_timer("OVERALL"):
        for i, obj in enumerate(train_dataset):
            print("{}: {}".format(i,obj["preprocessed_spectrogram"].shape))
            try:
                print("preprocessing_value: {}".format(obj["preprocessing_value"]))
            except:
                pass
        #progress_bar.update(1)
    #progress_bar.close()
    #print("mfccs : "+str(sample["mfccs"]))
    #print("chroma: "+str(sample["chroma"]))
    #print("mel: "+ str(sample["mel"]))
    #print("contrast: "+str(sample["contrast"]))
    #print("tonnetz: "+str(sample["tonnetz"]))

    print_code_stats()
    
    #TODO Disattivare drop silent


    