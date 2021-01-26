import librosa
import soundfile as sf
import os
import os.path
import random
import numpy as np

try:
    from utils.audio_utils import play_sound, load_audio_file
    from utils.timing import code_timer
except:
    pass

def write_audio_file(file, data, sample_rate):
        #librosa.output.write_wav(file, data, sample_rate)
        sf.write(file, data,sample_rate)

def load_file(name_file):
    y, sr = librosa.load(name_file)
    return y, sr

def pitch_shifting(y,sr,steps):
    y_changed = librosa.effects.pitch_shift(y, sr, n_steps=tone_step)
    return y_changed


class PitchShift(object):
    def __init__(self, values, debug_time = False):
        assert isinstance(values, list) and len(values)>0, "Please provide a list of possible pitch shifting semitones values to choose from (randomly)"
        self.values = values
        self.debug_time = debug_time
        self.name = "PitchShift"
        #print("Instance of PitchShift created")
    def __call__(self, y, sr=22050, value = None):
        
        if value is None:
            #with code_timer("PitchShift np.random.choice", debug=self.debug_time):
            value = np.random.choice(self.values)
        #with code_timer("PitchShift librosa", debug=self.debug_time):
        y_changed = librosa.effects.pitch_shift(y, sr, n_steps=value)
        return y_changed
    
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
        #print("Instance of TimeStretch created")
    def __call__(self, y, value = None):
        if value is None:
            #with code_timer("TimeStretch np.random.choice", debug=self.debug_time):
            value = np.random.choice(self.values)
        #with code_timer("TimeStretch librosa", debug=self.debug_time):
        y_changed = librosa.effects.time_stretch(y, value)
        return y_changed


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
class DynamicRangeCompression(object):
    def __init__(self,sound_file,min_dB,max_dB):
        self.sound_file = sound_file
        self.min_dB = min_dB
        self.max_dB = max_dB
        self.name = "DynamicRangeCompression"
    
    def __call__(self):
        #librosa.load return a time series representing amplitude not in db
        y1_db = librosa.amplitude_to_db(self.sound_file)
        #clip is done on decibel, but then must be translated back to amplitude
        t1_db = np.clip(y1_db, a_min = self.min_dB, a_max=self.max_dB)
        #back to amplitude
        t1 = librosa.db_to_amplitude(t1_db)

        return t1



class BackgroundNoise(object):
    def __init__(self, sound_file, loaded_audio_files):
        self.sound_file = sound_file
        self.loaded_audio_files = loaded_audio_files
        self.name = "BackgroundNoise"
    
    def __call__(self, index_file, weight = None, debug = False):
        #files : lista di Stringhe con i path dei noises
        #index : indice del noise scelto
        if weight is not None:
            assert weight <= 1.0 and weight >= 0.0
        else:
            weight = random.uniform(0.0, 0.5)
            if debug: print("--- weight random: ",weight)
        if debug: print("NOISE : ",self.loaded_audio_files[index_file])
        y1, sample_rate1 = load_audio_file(self.sound_file)
        y2, sample_rate2 = self.loaded_audio_files[index_file]
        y3 = ((1-weight)*y1 + weight*y2)/2
        sr=int((sample_rate1+sample_rate2)/2)

        if debug: play_sound(y3)
        return y3


if __name__ == "__main__":

    import sounddevice as sd

    import librosa
    import librosa.display
    import IPython as ip

    from os import listdir
    from os.path import isfile, join

    def play_sound(sound, sr = 22050, blocking=True):
        sd.play(sound, sr, blocking=True)

        
    def load_audio_file(path, duration = 4000, sample_rate = 22050, fixed_length = 88200):
        data, sample_rate = librosa.load(path, sr=sample_rate, mono=True,  dtype=np.float32)
        if len(data)>fixed_length:
            data = data[:fixed_length]
        else:
            data = np.concatenate((data, np.zeros(int(fixed_length - len(data)))))
        return data, sample_rate
    
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

    bn = BackgroundNoise(sound_file,loaded_audio_files)

    background_noise = bn(2,0.5)
    play_sound(background_noise)
    print(background_noise)
    
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
    


