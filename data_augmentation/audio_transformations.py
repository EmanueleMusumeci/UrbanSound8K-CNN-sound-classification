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
        preprocessed_clip = librosa.effects.time_stretch(clip, value)
        return preprocessed_clip


#DONT delete this
"""
Parameterizations from Dolby E Standard:

- Music Standard
    Max Boost: 12 dB (below -55 dB)
    Boost Range: -55 dB to -31 dB (2:1 ratio)//guadagno
    Null Band Width: 5 dB (-31 dB to -26 dB)
    Early Cut Range: -26 dB to -16 dB (2:1 ratio)
    Cut Range: -16 dB to +4 dB (20:1 ratio) 

- Film Standard
    Max Boost: 6 dB (below -43 dB)
    Boost Range: -43 dB to -31 dB (2:1 ratio)
    Null Band Width: 5 dB (-31 dB to -26 dB)
    Early Cut Range: -26 dB to -16 dB (2:1 ratio)
    Cut Range: -16 dB to +4 dB (20:1 ratio) 

- Speech
    Max Boost: 15 dB (below -50 dB)
    Boost Range: -50 dB to -31 dB (5:1 ratio)
    Null Band Width: 5 dB (-31 dB to -26 dB)
    Early Cut Range: -26 dB to -16 dB (2:1 ratio)
    Cut Range: -16 dB to +4 dB (20:1 ratio) 

- Film Light
    Max Boost: 6 dB (below -53 dB)
    Boost Range: -53 dB to -41 dB (2:1 ratio)
    Null Band Width: 20 dB (-41 dB to -21 dB)
    Early Cut Range: -26 dB to -11 dB (2:1 ratio)
    Cut Range: -11 dB to +4 dB (20:1 ratio) 

Parameterizations from icecast: see https://icecast.imux.net/viewtopic.php?t=3462


"""
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
    """
    ############################################################################################## test PitchShift

    print("-------- original sound")

    play_sound(y1)
    #( in article :steps (semitones) PS1 = {-2,-1,1,2} , PS2 = {-3.5,-2.5, 2.5,3.5} )
    pitch_shifting = PitchShift([-3.5, -2.5, 2.5, 3.5])
    audio_with_pitch_shifting = pitch_shifting(y1,sample_rate1)
    print("-------- PitchShifting:")
    play_sound(audio_with_pitch_shifting)
    
    
    ############################################################################################## test TimeStretching


    #set to generate the corresponding stretch  ( in article: {0.81, 0.93, 1.07, 1.23})
    #stretching_factor = 1.07
    print("-------- original sound")
    play_sound(y1)
    time_stretching = TimeStretch([0.81, 0.93, 1.07, 1.23])
    audio_with_time_stretching = time_stretching(y1)
    print("-------- TimeStretching:")
    play_sound(audio_with_time_stretching)
    """
    ############################################################################################## test BackGroundNoise

    noise_file = noise_file.replace("data_augmentation\\","")

    y1, sample_rate1 = load_audio_file(sound_file)
    print("-------- original sound")
    play_sound(y1)
    #lista dei file noises TODO Michele funzione più ordinata
    base_dir = base_dir.replace("data_augmentation","")
    noises_path = os.path.join(base_dir,"data","UrbanSound8K-JAMS","background_noise")
    onlyfiles = [f for f in listdir(noises_path) if isfile(join(noises_path, f))]
    onlyfiles = onlyfiles[:-1]
    new_only_files = []
    for i in onlyfiles:
        i = noises_path+ "\\" + i
        new_only_files.append(i)
    #print("noises files: ",new_only_files)
    
    loaded_audio_files = []
    for i in range(4):
        loaded_audio_files.append(load_audio_file(new_only_files[i]))
    
    print("-------- BackGroundNoise:")

    bn = BackgroundNoise(sound_file=sound_file,loaded_audio_files=new_only_files,index_file=2)

    background_noise = bn(0.5)
    play_sound(background_noise)
    print(background_noise)
    """
    ############################################################################################## test DRC

    
    y1, sample_rate1 = load_audio_file(sound_file)
    print("-------- original sound")
    play_sound(y1)
    #Music Standard  -> Max Boost: 12 dB (below -55 dB)
    drc_music_standard = DynamicRangeCompression(y1,min_dB = -55, max_dB = 12)

    #Film Standard -> Max Boost: 6 dB (below -43 dB)
    drc_film_standard = DynamicRangeCompression(y1, min_dB = -43, max_dB = 6)

    #Speech Standard   -> Max Boost: 15 dB (below -50 dB)
    drc_speech_standard = DynamicRangeCompression(y1,min_dB = -50, max_dB = 15)
    print("-------- DRC:")
    drc1 = drc_music_standard()
    play_sound(drc1)
    print("drc1: ",drc1)
    drc2 = drc_film_standard()
    play_sound(drc2)
    print("drc2: ",drc2)
    drc3 = drc_speech_standard()
    play_sound(drc3)
    print("drc3: ",drc3)
    """
    ############################################################################################## test DRC

    y1, sample_rate1 = load_audio_file(sound_file)
    print("-------- original sound")
    play_sound(y1)
    print(y1)
    """
    import muda.deformers
    #y_out = __sox(y1,sample_rate1,["0.1,0.3", "-90,-90,-70,-55,-50,-35,-31,-31,-21,-21,0,-20", "0", "0", "0.1"])
    #drc = muda.deformers.DynamicRangeCompression(preset=['film standard'])
    #print(drc)

    y_out = drc(y1,sample_rate1,'film standard')
    """

    y_out = drc(y1,sample_rate1,'film standard')
    print("------- film standard")
    play_sound(y_out)
    plot_periodogram(y_out,show=True)

    print(y_out)

    y_out = drc(y1,sample_rate1,'speech')
    print("------- speech")
    play_sound(y_out)
    plot_periodogram(y_out,show=True)

    print(y_out)

    
    y_out = drc(y1,sample_rate1,'music standard')
    print("------- music standard")
    play_sound(y_out)
    plot_periodogram(y_out,show=True)

    print(y_out)


    y_out = drc(y1,sample_rate1,'radio')
    print("------- radio")
    play_sound(y_out)
    plot_periodogram(y_out,show=True)
    print(y_out)

