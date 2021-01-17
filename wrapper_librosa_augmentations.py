import librosa
import soundfile as sf
import os



def write_audio_file(file, data, sample_rate):
        #librosa.output.write_wav(file, data, sample_rate)
        sf.write(file, data,sample_rate)

def load_file(name_file):
    y, sr = librosa.load(name_file)
    return y, sr

def pitch_shifting(y,sr,steps):
    y_changed = librosa.effects.pitch_shift(y, sr, n_steps=tone_step)
    return y_changed


class PitchShifting(object):
    def __init__(self):
        print("instance of PitchShifting created")
    def __call__(self,y,sr,semitones):
        y_changed = librosa.effects.pitch_shift(y, sr, n_steps=semitones)
        return y_changed


class TimeStretching(object):
    def __init__(self):
        print("instance of TimeStretching created")
    def __call__(self,y,factor):
        y_changed = librosa.effects.time_stretch(y, factor)
        return y_changed

#TODO Michele
#DRC
#BG


#set it to generate the corresponding shift ( in article :steps (semitones) PS1 = {-2,-1,1,2} , PS2 = {-3.5,-2.5, 2.5,3.5} )
semitone = 3.5
#set to generate the corresponding stretch  ( in article: {0.81, 0.93, 1.07, 1.23})
stretching_factor = 1.07
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.realpath(__file__))
    DATASET_DIR = os.path.join(base_dir,"data")
    print(librosa.__version__)

    name_file = "7061-6-0-0"
    type_file = ".wav"
    
    file_to_test = os.path.join(DATASET_DIR,name_file+type_file)
    print(file_to_test)

    y, sr = load_file(file_to_test)
    
    aug = os.path.join(DATASET_DIR,"augmentation")
    if not os.path.exists(aug): os.mkdir(aug)
    #test PitchShifting

    ps = os.path.join(aug, "pitch")
    if not os.path.exists(ps): os.mkdir(ps)


    pitch_shifting = PitchShifting()

    audio_with_pitch_shifting = pitch_shifting(y,sr,semitone)

    file_to_write = os.path.join(ps,name_file+"_"+str(semitone)+"sem"+type_file)
    write_audio_file(file_to_write,audio_with_pitch_shifting,sr)
    
    #test TimeStretching

    ts = os.path.join(aug, "time_stretching")
    if not os.path.exists(ts): os.mkdir(ts)
    
    time_stretching = TimeStretching()

    audio_with_time_stretching = time_stretching(y,stretching_factor)

    file_to_write = os.path.join(ts,name_file+"_factor_"+str(stretching_factor)+type_file)
    write_audio_file(file_to_write,audio_with_time_stretching,sr)
    
    



