import math

import librosa
import pydub
from pydub.utils import which
#This should fix the ffmpeg decoding errors as in https://github.com/jiaaro/pydub/issues/173
pydub.AudioSegment.converter = which("ffmpeg")

import numpy as np

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

def normalize_clip(audio_clip):
    normalization_factor = 1 / np.max(np.abs(audio_clip)) 
    audio_clip = audio_clip * normalization_factor
    return audio_clip

class MultipleWindowSelector:
    def __init__(self, window_size_seconds, sampling_rate, overlap=None, spectrogram_hop_length = None, random_location=False, drop_last = True):
        self.window_size_seconds = window_size_seconds
        self.sampling_rate = sampling_rate
        
        self.window_size = window_size_seconds * sampling_rate

        if overlap is not None:
            assert isinstance(float, overlap) and overlap > 0 and overlap < 1, "overlap should be a number between 0 and 1"
            self.step_size = int(math.floor(self.window_size * (1-overlap)))
        else:
            self.step_size = self.window_size

        self.spectrogram_hop_length = spectrogram_hop_length
        if self.spectrogram_hop_length is not None:
            self.spectrogram_window_size = int(math.ceil(self.window_size/spectrogram_hop_length))
            if overlap is not None:
                self.spectrogram_step_size = int(math.ceil(self.spectrogram_window_size * (1-overlap)))

        self.drop_last = drop_last

        #If random_location is True, the step size will be chosen randomly using the computed step size as an upper bound
        self.random_location = random_location
    
    def __call__(self, clip, spectrogram=None):
        assert len(clip) >= self.window_size, "Window size ({}) is bigger than audio clip length ({})".format(self.window_size, len(clip))
        total_clip_frames = len(clip)
        total_spectrogram_frames = len(spectrogram)
        begin = 0
        while begin < total_clip_frames:
            if spectrogram is not None:
                assert self.spectrogram_hop_length is not None, "Please specify a hop length for the spectrogram"
                spectrogram_begin = int(math.floor(begin/self.spectrogram_hop_length))
                yield (begin, begin + self.window_size, spectrogram_begin, spectrogram_begin + self.spectrogram_window_size)
            else:
                yield (begin, begin + self.window_size)
            if self.random_location:
                random_value = np.random.rand()
                #compute random step_size (remapping 0 to 1 in case the random value is 0)
                random_step_size = int(math.floor((random_value if random_value>0 else 1) * self.step_size))
                begin += random_step_size
            else:
                begin += self.step_size
        #If true, the last segment will be dropped if its length is lower than the segment size
        if not self.drop_last:
            if spectrogram is not None:
                yield (begin, total_clip_frames, spectrogram_begin, total_spectrogram_frames)
            else:
                yield (begin, total_clip_frames)


class SingleWindowSelector:
    #if random_location is False, selects a window from the beginning of the clip,
    #else selects from a random location
    def __init__(self, window_size_seconds, sampling_rate = 22050, spectrogram_hop_length=None, random_location = True):
        self.window_size_seconds = window_size_seconds
        self.sampling_rate = sampling_rate

        self.window_size = window_size_seconds * sampling_rate

        self.spectrogram_hop_length = spectrogram_hop_length
        if self.spectrogram_hop_length is not None:
            self.spectrogram_window_size = int(math.ceil(self.window_size/spectrogram_hop_length))

        self.random_location = random_location

    def __call__(self, clip, spectrogram=None):
        assert len(clip) >= self.window_size, "Window size ({}) is bigger than audio clip length ({})".format(self.window_size, len(clip))
        if self.random_location:
            begin = np.random.randint(0, len(clip)-self.window_size)
        else:
            begin = 0
        if spectrogram is not None:
            assert self.spectrogram_hop_length is not None, "Please specify a hop length for the spectrogram"
            spectrogram_begin = int(math.floor(begin/self.spectrogram_hop_length))
            yield (begin, begin+self.window_size, spectrogram_begin, spectrogram_begin+self.spectrogram_window_size)
        else:
            yield (begin, begin+self.window_size)
