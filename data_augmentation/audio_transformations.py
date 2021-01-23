import librosa
import soundfile as sf
import os

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
        #print("Instance of PitchShift created")
    def __call__(self,y,sr=22050,semitones=None):
        
        if semitones is None:
            with code_timer("PitchShift np.random.choice", debug=self.debug_time):
                semitones = np.random.choice(self.values)
        with code_timer("PitchShift librosa", debug=self.debug_time):
            y_changed = librosa.effects.pitch_shift(y, sr, n_steps=semitones)
        return y_changed


class TimeStretch(object):
    def __init__(self, values, debug_time = False):
        assert isinstance(values, list) and len(values)>0, "Please provide a list of possible stretching factors to choose from (randomly)"
        self.values = values
        self.debug_time = debug_time
        #print("Instance of TimeStretch created")
    def __call__(self,y, factor = None):
        if factor is None:
            with code_timer("TimeStretch np.random.choice", debug=self.debug_time):
                factor = np.random.choice(self.values)
        with code_timer("TimeStretch librosa", debug=self.debug_time):
            y_changed = librosa.effects.time_stretch(y, factor)
        return y_changed

#TODO Michele
#DRC
#BG



if __name__ == "__main__":
    
    base_dir = os.path.dirname(os.path.realpath(__file__))
    DATASET_DIR = os.path.join(base_dir,"data")
    print(librosa.__version__)

    name_file = "7061-6-0-0"
    type_file = ".wav"
    
    file_to_test = os.path.join(DATASET_DIR,"UrbanSound8K","audio","fold1",name_file+type_file)
    print(file_to_test)

    #y, sr = load_audio_file(file_to_test)
    
    #aug = os.path.join(DATASET_DIR,"augmentation")
    #if not os.path.exists(aug): os.mkdir(aug)
    #test PitchShifting

    #ps = os.path.join(aug, "pitch")
    #if not os.path.exists(ps): os.mkdir(ps)


    #set it to generate the corresponding shift ( in article :steps (semitones) PS1 = {-2,-1,1,2} , PS2 = {-3.5,-2.5, 2.5,3.5} )
    #semitone = 3.5
    """
    play_sound(y)
    pitch_shifting = PitchShift([-3.5, -2.5, 2.5, 3.5])
    audio_with_pitch_shifting = pitch_shifting(y,sr)
    play_sound(audio_with_pitch_shifting)
    """
    #file_to_write = os.path.join(ps,name_file+"_"+str(semitone)+"sem"+type_file)
    #write_audio_file(file_to_write,audio_with_pitch_shifting,sr)
    
    #test TimeStretching

    #ts = os.path.join(aug, "time_stretching")
    #if not os.path.exists(ts): os.mkdir(ts)
    
    #set to generate the corresponding stretch  ( in article: {0.81, 0.93, 1.07, 1.23})
    #stretching_factor = 1.07
    """
    play_sound(y)
    time_stretching = TimeStretch([0.81, 0.93, 1.07, 1.23])
    audio_with_time_stretching = time_stretching(y)
    play_sound(audio_with_time_stretching)
    """
    #file_to_write = os.path.join(ts,name_file+"_factor_"+str(stretching_factor)+type_file)
    #write_audio_file(file_to_write,audio_with_time_stretching,sr)

    #test background noise from stackoverflow:
    #   https://stackoverflow.com/questions/4039158/mixing-two-audio-files-together-with-python
    
    #import numpy as np
    #from scikits.audiolab import wavread

   
    #from pydub import AudioSegment
    #file_to_test.replace("data_augmentation\\","")
    file_to_test = file_to_test.replace("data_augmentation\\","")

    import librosa
    import librosa.display
    import IPython as ip

    y1, sample_rate1 = librosa.load(file_to_test, mono=True)
    y2, sample_rate2 = librosa.load(file_to_test, mono=True)

    y3 = (y1+y2)/2

    sr=int((sample_rate1+sample_rate2)/2)

    play_sound(y3)
    
   


    



