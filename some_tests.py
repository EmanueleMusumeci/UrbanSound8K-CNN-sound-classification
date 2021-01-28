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
from utils.plot_utils import plot_periodogram, plot_sound_waves
from utils.audio_utils import load_audio_file, play_sound
from utils.spectrogram_utils import generate_mel_spectrogram_librosa, display_heatmap

from data_augmentation.audio_transformations import *

import scipy.signal

if __name__ == "__main__":

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
    #play_sound(y1)
    display_heatmap(generate_mel_spectrogram_librosa(y1, 128))

    background_noise = BackgroundNoise({
        "street_scene_1" : "150993__saphe__street-scene-1.wav",
        #"street_scene_3" : "173955__saphe__street-scene-3.wav",
        #"street_valencia" : "207208__jormarp__high-street-of-gandia-valencia-spain.wav",
        #"city_park_tel_aviv" : "268903__yonts__city-park-tel-aviv-israel.wav",
    }, files_dir = os.path.join(DATASET_DIR, "UrbanSound8K-JAMS", "background_noise"))

    preprocessed_y1 = background_noise(y1, play=True)
    #play_sound(preprocessed_y1)
    display_heatmap(generate_mel_spectrogram_librosa(preprocessed_y1, 128))
    

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
    base_dir = os.path.dirname(os.path.realpath(__file__))

    save_dir = "data"
    y1, sample_rate1 = load_audio_file(sound_file)
    print("-------- original sound")
    #play_sound(y1)
    print(y1)
    file_output = os.path.join(base_dir,save_dir)
    #file_output = os.path.join(file_output,"original")
    plot_sound_waves(y1,sound_file_name="original",show=True,save_to_dir=file_output)

    
    y_out = drc(y1,sample_rate1,'film standard')
    print("------- film standard")
    #play_sound(y_out)
    plot_sound_waves(y_out,sound_file_name="film standard",show=True,save_to_dir=file_output)


    print(y_out)

    y_out = drc(y1,sample_rate1,'speech')
    print("------- speech")
    #play_sound(y_out)
    plot_sound_waves(y_out,sound_file_name="speech",show=True,save_to_dir=file_output)


    print(y_out)

    
    y_out = drc(y1,sample_rate1,'music standard')
    print("------- music standard")
    #play_sound(y_out)
    plot_sound_waves(y_out,sound_file_name="music standard",show=True,save_to_dir=file_output)

    print(y_out)


    y_out = drc(y1,sample_rate1,'radio')
    print("------- radio")
    #play_sound(y_out)
    plot_sound_waves(y_out,sound_file_name="radio",show=True,save_to_dir=file_output)

    print(y_out)
