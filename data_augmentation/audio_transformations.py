import os
import random
import copy

import math

import librosa
import soundfile as sf

import numpy as np

from utils.audio_utils import play_sound, load_audio_file
from utils.timing import code_timer

import muda

class PitchShift(object):
    def __init__(self, values, debug_time = False, name = "PitchShift"):
        '''
        Implements the PitchShift transformation 
        Args:
          - values: iterable dataset that provides the samples
          OPTIONAL:
            - debug_time: boolean to set debug mode
        '''
        assert isinstance(values, list) and len(values)>0, "Please provide a list of possible pitch shifting semitones values to choose from (randomly)"
        self.values = values
        self.debug_time = debug_time
        self.name = name

        
    def __call__(self, clip, sr=22050, value = None):
        '''
        Implements the PitchShift transformation wrapping on librosa
        Args:
          - clip: clip on which apply transformation
          OPTIONAL:
            - sr: sampling rate of the clip
            - value: numbers of semitones
        '''
        if value is None:
            #with code_timer("PitchShift np.random.choice", debug=self.debug_time):
            value = np.random.choice(self.values)
        #with code_timer("PitchShift librosa", debug=self.debug_time):
        preprocessed_clip = librosa.effects.pitch_shift(clip, sr, n_steps=value)
        return preprocessed_clip
    
    def get_value_labels(self):
        labels = []
        for value in self.values:
            labels.append(str(value))
        return labels

class TimeStretch(object):
    def __init__(self, values, debug_time = False, name = "TimeStretch"):
        '''
        Implements the TimeStretch transformation 
        Args:
          - values: iterable dataset that provides the samples
          OPTIONAL:
            - debug_time: boolean to set debug mode
        '''
        assert isinstance(values, list) and len(values)>0, "Please provide a list of possible stretching factors to choose from (randomly)"
        self.values = values
        self.debug_time = debug_time
        self.name = name

    def __call__(self, clip, value = None):
        '''
        Implements the TimeStretch transformation wrapping on librosa
        Args:
          - clip: clip on which apply transformation
          OPTIONAL:
            - value: numbers of semitones
        '''
        if value is None:
            #with code_timer("TimeStretch np.random.choice", debug=self.debug_time):
            value = np.random.choice(self.values)
        #with code_timer("TimeStretch librosa", debug=self.debug_time):
        preprocessed_clip = librosa.effects.time_stretch(clip, value)
        return preprocessed_clip



class MUDADynamicRangeCompression(object):
    def __init__(self, sample_rate = 22050, name = "DynamicRangeCompression"):
        '''
        Implements the Dynamic Range Compression transformation 
        Args:
          OPTIONAL:
            - sample_rate: sample rate of the clip
        '''
        self.values = list(muda.deformers.PRESETS.keys())

        self.sample_rate = sample_rate
        '''
        assert isinstance(presets, dict) or isinstance(presets, str), "presets should be a dictionary or a directory to a JSON dictionary containing Dolby Digital DRC presets"
        if isinstance(presets, dict):
            self.presets = presets
        elif isinstance(presets, str):
            with open(presets, "r") as f:
                self.presets = json.load(f)
        '''
        
        self.name = name
    
    def get_value_labels(self):
        return copy.deepcopy(self.values)

    def __call__(self, sound, value = None, sample_rate = 22050, debug = False):
        '''
        Implements the Dynamic Range Compression transformation  wrapping on muda
        Args:
          - sound: clip on which apply transformation
          OPTIONAL:
            - value:
            - sample_rate: sample rate of the clip
            - debug: boolean to set debug mode
            
        '''
        if value is None:
            value = self.values[np.random.randint(low = 0, high = len(self.values))]
        
        if debug: print("Chosen preset: "+value)

        preprocessed_clip = muda.deformers.sox.drc(sound, self.sample_rate, value)
        
        return preprocessed_clip

class BackgroundNoise(object):
    def __init__(self, background_files, files_dir, name = "BackgroundNoise"):
        '''
        Implements the BackgroundNoise transformation 
        Args:
          - background_files: .wav file of bacground noises
          - files_dir: directory in which are the .wav files
        '''
        
        self.background_clips = {}
        self.values = []
        
        print("Loading background noise clips...")
        #Load background files
        for background_file_id, background_file_name in background_files.items():
            background_file_dir = os.path.join(files_dir, background_file_name)
            assert os.path.exists(background_file_dir), "Specified background file {} does not exist".format(background_file_dir)
            self.background_clips[background_file_id], _ = load_audio_file(background_file_dir)
            self.values.append(background_file_id)

        self.name = name
        print("Clips loaded")
    
    def get_value_labels(self):
        return copy.deepcopy(self.values)

    def __call__(self, sound, value = None, debug = False, play = False, weight = None):
        '''
        Implements the BackgroundNoise transformation  wrapping on librosa
        Args:
          - sound: clip on which apply transformation
          OPTIONAL:
            - value:
            - debug: boolean to set debug mode
            - play: True is you want play the sound
            - weight : value in range [0.0, 0.5]

            
        '''
        if value is None:
            value = self.values[np.random.randint(low = 0, high = len(self.values))]

        #Choose random overlaying weight in range [0.1, 0.5] distributed uniformly
        if weight is not None:
            assert weight >= 0.1 and weight <= 0.5, "Weight should be in range [0, 1]"
        else:
            weight = np.random.uniform(low = 0.1, high = 0.5)

        if debug: print("background_file_id", value)

        #Load background clip
        bg_noise = self.background_clips[value]
        
        #Even out clip and background clip lengths
        if len(bg_noise)>len(sound):
            #If clip is shorter than bg_noise, choose random clip from bg_noise of the same length
            clip_length = len(sound)
            bg_noise_begin = np.random.randint(0, high=len(bg_noise)-clip_length)
            bg_noise_end = bg_noise_begin + clip_length
            bg_noise = bg_noise[bg_noise_begin: bg_noise_end]
        elif len(bg_noise)<len(sound):
            #If clip is longer than bg_noise, repeat bg_noise to match clip length
            n_repeats = math.floor(len(sound) / len(bg_noise))
            remaining = len(sound) - len(bg_noise) * n_repeats
            bg_noise = np.repeat(bg_noise, n_repeats)
            bg_noise = np.concatenate(bg_noise, bg_noise[:remaining])

        #Overlay the two clips
        preprocessed_clip = ((1-weight)*sound + weight*bg_noise)/2

        if play:
            print("Playing original clip")
            play_sound(sound)
            print("Playing background noise")
            play_sound(preprocessed_clip)
            print("Playing preprocessed clip")

        return preprocessed_clip

