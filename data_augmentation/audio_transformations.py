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
    def __init__(self, values, debug_time = False):
        assert isinstance(values, list) and len(values)>0, "Please provide a list of possible pitch shifting semitones values to choose from (randomly)"
        self.values = values
        self.debug_time = debug_time
        self.name = "PitchShift"
        
    def __call__(self, clip, sr=22050, value = None):
        
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
    def __init__(self, values, debug_time = False):
        assert isinstance(values, list) and len(values)>0, "Please provide a list of possible stretching factors to choose from (randomly)"
        self.values = values
        self.debug_time = debug_time
        self.name = "TimeStretch"

    def __call__(self, clip, value = None):
        if value is None:
            #with code_timer("TimeStretch np.random.choice", debug=self.debug_time):
            value = np.random.choice(self.values)
        #with code_timer("TimeStretch librosa", debug=self.debug_time):
        preprocessed_clip = librosa.effects.time_stretch(y, value)
        return preprocessed_clip



class MUDADynamicRangeCompression(object):
    def __init__(self, sample_rate = 22050):
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
        
        self.name = "DynamicRangeCompression"
    
    def get_value_labels(self):
        return copy.deepcopy(self.values)

    def __call__(self, sound, value = None, sample_rate = 22050, debug = False):
        if value is None:
            value = self.values[np.random.randint(low = 0, high = len(self.values))]
        
        if debug: print("Chosen preset: "+value)

        preprocessed_clip = muda.deformers.sox.drc(sound, self.sample_rate, value)
        
        return preprocessed_clip

class BackgroundNoise(object):
    def __init__(self, background_files, files_dir):
        
        self.background_clips = {}
        self.values = []
        
        print("Loading background noise clips...")
        #Load background files
        for background_file_id, background_file_name in background_files.items():
            background_file_dir = os.path.join(files_dir, background_file_name)
            assert os.path.exists(background_file_dir), "Specified background file {} does not exist".format(background_file_dir)
            self.background_clips[background_file_id], _ = load_audio_file(background_file_dir)
            self.values.append(background_file_id)

        self.name = "BackgroundNoise"
        print("Clips loaded")
    
    def get_value_labels(self):
        return copy.deepcopy(self.values)

    def __call__(self, sound, value = None, debug = False, play = False, weight = None):
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

        #sr=int((sample_rate1+sample_rate2)/2)

        if play:
            print("Playing original clip")
            play_sound(sound)
            print("Playing background noise")
            play_sound(preprocessed_clip)
            print("Playing preprocessed clip")

        return preprocessed_clip

if __name__ == "__main__":

    import sounddevice as sd

    import librosa
    import librosa.display
    import IPython as ip

    from os import listdir
    from os.path import isfile, join
    
    base_dir = os.path.dirname(os.path.realpath(__file__))
    DATASET_DIR = os.path.join(base_dir,"data")
    print(librosa.__version__)

    name_file = "7061-6-0-0"
    type_file = ".wav"
    
    sound_file = os.path.join(DATASET_DIR,"UrbanSound8K","audio","fold1",name_file+type_file)
    noise_file = os.path.join(DATASET_DIR,"UrbanSound8K-JAMS","background_noise","150993__saphe__street-scene-1.wav")
    print("sound_file: ",sound_file)
    print("noise_file: ",noise_file)

    #set it to generate the corresponding shift ( in article :steps (semitones) PS1 = {-2,-1,1,2} , PS2 = {-3.5,-2.5, 2.5,3.5} )
    sound_file = sound_file.replace("data_augmentation\\","")
    y1, sample_rate1 = load_audio_file(sound_file)
    