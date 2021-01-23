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

    from utils.dataset_utils import load_compacted_dataset
    from utils.audio_utils import *
    from utils.spectrogram_utils import *
    from utils.timing import *
except:
    pass


class SoundDatasetFold(torch.utils.data.IterableDataset):
    def __init__(self, dataset_dir, dataset_name, 
                folds = [], 
                shuffle = False, 
                
                selected_classes = [0,1,2,3,4,5,6,7,8,9],

                use_spectrograms = True, 

                audio_segment_selector = None,

                shift_transformation = None,
                background_noise_transformation = None,
                
                time_stretch_transformation = None,
                pitch_shift_transformation = None,

                audio_clip_duration = 4000,
                sample_rate = 22050,

                spectrogram_bands = 60,
                spectrogram_hop_length = 512,
                spectrogram_frames_per_segment = 41,
                spectrogram_window_overlap = 0.5,
                drop_last_spectrogram_frame = True,
                normalize_audio = True,
                silent_clip_cutoff_dB = -70,

                compute_deltas = True,
                compute_delta_deltas = False,
                test = False,
                progress_bar = False,
                debug_preprocessing_time = False,

                select_percentage_of_dataset = 1.0,

                load_compacted = True
                ):
        super(SoundDatasetFold).__init__()
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        self.folds = folds

        self.selected_classes = selected_classes
        self.num_classes = len(selected_classes)

        self.test_mode = test

        self.load_compacted = load_compacted
        if self.load_compacted:
            #Load from compacted dataset files, returning only sample_ids whose class id is among the selected ones (so the dataset will ignore the other ones)
            self.audio_meta, self.audio_raw, self.audio_spectrograms = load_compacted_dataset(dataset_dir, folds=self.folds)
        else:
            #Load dataset index (the audio files will be loaded one by one therefore training will be up to 10x slower!!!), 
            # returning only the sample_ids whose class_id is among the selected ones
            self.audio_meta = self.load_dataset_index(dataset_dir, folds=self.folds)
        
        self.sample_ids = self.select_sample_ids(self.selected_classes, select_percentage = select_percentage_of_dataset)        

        self.shuffle = shuffle

        self.use_spectrograms = use_spectrograms

        self.shift_transformation = shift_transformation
        self.background_noise_transformation = background_noise_transformation

        self.time_stretch_transformation = time_stretch_transformation
        self.pitch_shift_transformation = pitch_shift_transformation

        self.audio_clip_duration = audio_clip_duration
        self.sample_rate = sample_rate
        
        #as in https://github.com/karolpiczak/paper-2015-esc-convnet/blob/master/Code/_Datasets/Setup.ipynb
        #self.spectrogram_frames_per_segment = spectrogram_frames_per_segment
        #self.spectrogram_hop_length = spectrogram_hop_length
        #self.spectrogram_window_size = self.spectrogram_hop_length * (spectrogram_frames_per_segment-1)
        #assert spectrogram_window_overlap > 0 and spectrogram_window_overlap < 1, "spectrogram_window_overlap should be between 0 and 1"
        #self.spectrogram_window_overlap = spectrogram_window_overlap
        #self.spectrogram_window_step_size = math.floor(self.spectrogram_window_size * (1-spectrogram_window_overlap))
        #self.spectrogram_bands = spectrogram_bands
        #self.drop_last_spectrogram_frame = drop_last_spectrogram_frame

        #If no audio segment selector is specified, a SingleWindowSelector will be constructed to return the whole clip 
        if audio_segment_selector is None:
            self.audio_segment_selector = SingleWindowSelector(4000, random_location=False)
        else:
            assert isinstance(audio_segment_selector, SingleWindowSelector) or isinstance(audio_segment_selector, MultipleWindowSelector), \
                "The audio_segment_selector should be a utils.audio_utils.SingleWindowSelector or utils.audio_utils.MultipleWindowSelector"
            self.audio_segment_selector = audio_segment_selector

        self.silent_clip_cutoff_dB = silent_clip_cutoff_dB

        self.normalize_audio = normalize_audio

        self.compute_deltas = compute_deltas
        self.compute_delta_deltas = compute_delta_deltas
    
        self.progress_bar = progress_bar

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
            return ["original_spectrogram", "preprocessed_spectrogram"]
        else:
            return ["mfccs", "chroma", "mel", "contrast", "tonnetz"]

    def get_unpreprocessed_fields(self): return ["class_name", "meta_data"]
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
        if self.load_compacted:
            sound = self.audio_raw[index]
        else:
            try:
                sound, sample_rate = load_audio_file(sample["file_path"], sample_rate=self.sample_rate)
            except:
                raise e

        compacted_spectrogram = None
        if self.audio_spectrograms is not None:
            compacted_spectrogram = self.audio_spectrograms[index]

        if self.use_spectrograms:
            original_spectrograms, preprocessed_spectrograms = self.preprocess_convolutional(sound, original_log_mel_spectrogram=compacted_spectrogram)
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
            mfccs, chroma, mel, contrast, tonnetz = self.preprocess_feed_forward(sound)
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

    #@function_timer
    def preprocess_convolutional(self, audio_clip, debug = False, original_log_mel_spectrogram = None):
        
        if self.normalize_audio:
            audio_clip = normalize_clip(audio_clip)
        
        if self.time_stretch_transformation is not None:
            if not self.test_mode:
                with code_timer("audio time stretch", debug=self.debug_preprocessing_time):
                    audio_clip = self.time_stretch_transformation(audio_clip)
                
        original_spectrograms = []
        preprocessed_spectrograms = []
        
        if debug: print("segments: "+str(list(self.audio_segment_selector(self.spectrogram_window_step_size, self.spectrogram_window_size, len(audio_clip), self.drop_last_spectrogram_frame))))
        
        for i, (segment_start, segment_end) in enumerate(self.audio_segment_selector(self.spectrogram_window_step_size, self.spectrogram_window_size, len(audio_clip), self.drop_last_spectrogram_frame)):
            original_signal_segment = audio_clip[segment_start:segment_end]
            #Only accept audio clip segments that are:
            #1) Of a fixed window size (self.spectrogram_window_size)
            #2) Fully contained in the audio clip (segments that go "out of bounds" wrt the audio clip are not considered)
            if len(original_signal_segment) == math.floor(self.spectrogram_window_size) and segment_end<len(audio_clip): 
                if debug: print("segment ("+str(segment_start)+", "+str(segment_end)+")")

                if original_log_mel_spectrogram is None:
                    original_log_mel_spectrogram = generate_mel_spectrogram_librosa(original_signal_segment, self.spectrogram_bands, debug_time_label=("original" if self.debug_preprocessing_time else ""), show=self.debug_spectrograms)
                #original_mel_spectrogram_essentia = generate_mel_spectrogram_essentia(original_signal_segment, self.spectrogram_bands, self.sample_rate, debug_time_label=("original" if self.debug_preprocessing_time else None))
                #display_heatmap(original_mel_spectrogram_librosa)
                #display_heatmap(original_mel_spectrogram_essentia)
                #raise
                with code_timer("drop silent", debug=self.debug_preprocessing_time):
                    #drop silent frames (taken from https://github.com/karolpiczak/paper-2015-esc-convnet/blob/master/Code/_Datasets/Setup.ipynb)
                    #ONLY if they aren't the only frame of the audio clip
                    if debug: print("Mean dB value:" +str(np.mean(original_log_mel_spectrogram)))
                    if i>0 and np.mean(original_log_mel_spectrogram) <= self.silent_clip_cutoff_dB:
                        if debug: print("Silent segment dropped")
                        continue
                
                original_spectrograms.append(original_log_mel_spectrogram)

                #Apply all audio augmentations, in sequence
                preprocessed_signal_segment = original_signal_segment

                if self.pitch_shift_transformation is not None:
                    if not self.test_mode:
                        with code_timer("audio pitch shift", debug=self.debug_preprocessing_time):
                            preprocessed_signal_segment = self.pitch_shift_transformation(preprocessed_signal_segment)
                        
                preprocessed_log_mel_spectrogram = generate_mel_spectrogram_librosa(preprocessed_signal_segment, self.spectrogram_bands, debug_time_label=("preprocessed" if self.debug_preprocessing_time else ""), show=self.debug_spectrograms)
                #original_mel_spectrogram = generate_mel_spectrogram_essentia(preprocessed_signal_segment, self.spectrogram_bands, self.sample_rate, debug_time_label=("preprocessed" if self.debug_preprocessing_time else None))

                preprocessed_spectrograms.append(preprocessed_log_mel_spectrogram)
        
        with code_timer("original spectrogram reshape", debug=self.debug_preprocessing_time):
        #Reshape the spectrograms from [N_AUDIO_SEGMENT, N_BANDS, N_FRAMES] to [N_AUDIO_SEGMENT, N_BANDS, N_FRAMES, 1]
            original_spectrograms = np.asarray(original_spectrograms).reshape(len(original_spectrograms),self.spectrogram_bands,self.spectrogram_frames_per_segment,1)
        
        with code_timer("preprocessed spectrogram reshape", debug=self.debug_preprocessing_time):
            preprocessed_spectrograms = np.asarray(preprocessed_spectrograms).reshape(len(preprocessed_spectrograms),self.spectrogram_bands,self.spectrogram_frames_per_segment,1)
        
        with code_timer("deltas", debug=self.debug_preprocessing_time):
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


        with code_timer("image augmentation", debug=self.debug_preprocessing_time):
            if not self.test_mode:
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
    
    INSTANCE_NAME = "PROVA"
    BATCH_SIZE = 128
    USE_CNN = True
    APPLY_IMAGE_AUGMENTATIONS = True
    APPLY_AUDIO_AUGMENTATIONS = True

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    if APPLY_AUDIO_AUGMENTATIONS:
        random_pitch_shift = PitchShift([-3.5, -2.5, 2.5, 3.5])
        random_time_stretch = TimeStretch([0.81, 0.93, 1.07, 1.23])
    else:
        random_pitch_shift = None
        random_time_stretch = None

    train_dataset = SoundDatasetFold(DATASET_DIR, DATASET_NAME, 
                                folds = [1], 
                                shuffle = True, 
                                use_spectrograms = USE_CNN, 
                                shift_transformation = right_shift_transformation, 
                                background_noise_transformation = background_noise_transformation, 
                                time_stretch_transformation = random_time_stretch,
                                pitch_shift_transformation = random_pitch_shift, 
                                spectrogram_frames_per_segment = spectrogram_frames_per_segment, 
                                spectrogram_bands = spectrogram_bands, 
                                compute_deltas=True, 
                                compute_delta_deltas=True, 
                                test = False, 
                                progress_bar = True,
                                selected_classes=selected_classes,
                                select_percentage_of_dataset=DATASET_PERCENTAGE,
                                debug_preprocessing_time=True
                                )   

    test_dataset = SoundDatasetFold(DATASET_DIR, DATASET_NAME, 
                                folds = [2], 
                                shuffle = False, 
                                use_spectrograms = USE_CNN, 
                                spectrogram_frames_per_segment = spectrogram_frames_per_segment, 
                                spectrogram_bands = spectrogram_bands, 
                                compute_deltas=True, 
                                compute_delta_deltas=True, 
                                test = True, 
                                progress_bar = True,
                                selected_classes=selected_classes,
                                select_percentage_of_dataset=DATASET_PERCENTAGE
                                )

    #progress_bar = tqdm(total=len(dataset), desc="Sample", position=0)
    with code_timer("OVERALL"):
        for i, obj in enumerate(train_dataset):
        #    continue
            if i>25:
                break
        #progress_bar.update(1)
    #progress_bar.close()
    #print("mfccs : "+str(sample["mfccs"]))
    #print("chroma: "+str(sample["chroma"]))
    #print("mel: "+ str(sample["mel"]))
    #print("contrast: "+str(sample["contrast"]))
    #print("tonnetz: "+str(sample["tonnetz"]))

    print_code_stats()
    


    