import librosa
import pydub
from pydub.utils import which
#This should fix the ffmpeg decoding errors as in https://github.com/jiaaro/pydub/issues/173
pydub.AudioSegment.converter = which("ffmpeg")

import sounddevice as sd

def play_sound(sound, sr = 22050, blocking=True):
    sd.play(sound, sr, blocking=True)

'''
This function loads an audio file
'''
def load_audio_file(path, duration = 4000, sample_rate = 22050, fixed_length = 88200):
    data, sample_rate = librosa.load(path, sr=sample_rate, mono=True,  dtype=np.float32)
    if len(data)>fixed_length:
        data = data[:fixed_length]
    else:
        data = np.concatenate((data, np.zeros(int(fixed_length - len(data)))))
    return data, sample_rate
    
#Snippet taken from https://github.com/karolpiczak/paper-2015-esc-convnet/blob/master/Code/_Datasets/Setup.ipynb
'''
This function loads an audio file for a specified duration (default 4 secs), overlays on a silent audio segment 
(providing a native padding), resamples it a t 22050 Hz, sets it to mono and converts it to float
'''
#@function_timer
def _load_audio_file(path, duration = 4000, sample_rate = 22050):

    print(mediainfo(path)["sample_rate"])
    print(mediainfo(path))
    
    #with code_timer("pydub.AudioSegment.silent"):
    silence = pydub.AudioSegment.silent(duration=duration, frame_rate=sample_rate)
    
    try:
        #with code_timer("pydub.AudioSegment.from_file"):
        audio_segment = pydub.AudioSegment.from_file(path).set_frame_rate(sample_rate).set_channels(1)
    #Small fix for the infamous pydub encoder exception with ffmpeg decoder (https://github.com/jiaaro/pydub/issues/415)
    except pydub.exceptions.CouldntDecodeError:
        audio_segment = pydub.AudioSegment.from_file_using_temporary_files(path).set_frame_rate(sample_rate).set_channels(1)

    audio_segment = silence.overlay(audio_segment)

    raw = (np.frombuffer(audio_segment._data, dtype="int16") + 0.5) / (0x7FFF + 0.5)   # convert to float
    if audio_segment.duration_seconds != duration/1000 or len(raw) != duration/1000 * sample_rate:
        print("Audio segment {} has duration {} and array length {}".format(path, audio_segment.duration_seconds,len(raw)))

    return raw, sample_rate
