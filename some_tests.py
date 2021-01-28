import librosa
import soundfile as sf
import os
import os.path
import random
import numpy as np

import sounddevice as sd

import librosa
import librosa.display
import IPython as ip

from os import listdir
from os.path import isfile, join

from muda.deformers.sox import drc
from utils.plot_utils import plot_periodogram
from utils.plot_utils import plot_sound_waves
import scipy.signal
from muda.deformers.sox import drc


if __name__ == "__main__":

    
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
    """
    ############################################################################################## test DRC

    y1, sample_rate1 = load_audio_file(sound_file)
    print("-------- original sound")
    #play_sound(y1)
    print(y1)
    plot_sound_waves(y1,show=True)

    """
    import muda.deformers
    #y_out = __sox(y1,sample_rate1,["0.1,0.3", "-90,-90,-70,-55,-50,-35,-31,-31,-21,-21,0,-20", "0", "0", "0.1"])
    #drc = muda.deformers.DynamicRangeCompression(preset=['film standard'])
    #print(drc)

    y_out = drc(y1,sample_rate1,'film standard')
    """
 

    y_out = drc(y1,sample_rate1,'film standard')
    print("------- film standard")
    #play_sound(y_out)
    plot_sound_waves(y_out,show=True)

    print(y_out)

    y_out = drc(y1,sample_rate1,'speech')
    print("------- speech")
    #play_sound(y_out)
    plot_sound_waves(y_out,show=True)

    print(y_out)

    
    y_out = drc(y1,sample_rate1,'music standard')
    print("------- music standard")
    #play_sound(y_out)
    plot_sound_waves(y_out,show=True)

    print(y_out)


    y_out = drc(y1,sample_rate1,'radio')
    print("------- radio")
    #play_sound(y_out)
    plot_sound_waves(y_out,show=True)
    print(y_out)

