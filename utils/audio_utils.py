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

'''
Normalizes an audio clip
'''
def normalize_clip(audio_clip):
    normalization_factor = 1 / np.max(np.abs(audio_clip)) 
    audio_clip = audio_clip * normalization_factor
    return audio_clip

'''
NOT USED - NOT TESTED - Most probably not working
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

    #TODO fix case where cui total_clip_frames=None
    def __call__(self, total_clip_frames=None, total_spectrogram_frames=None):
        assert total_clip_frames is not None or total_spectrogram_frames is not None, "To generate segments at least one among the clip length or the spectrogram length should be provided"
        if total_clip_frames is not None:
            assert total_clip_frames >= self.window_size, "Window size ({}) is bigger than audio clip length ({})".format(self.window_size, total_clip_frames)
        begin = 0
        while begin < total_clip_frames:
            if total_spectrogram_frames is not None:
                assert self.spectrogram_hop_length is not None, "Specify a hop length for the spectrogram"
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
            if total_spectrogram_frames is not None:
                if total_clip_frames is not None:
                    yield (begin, total_clip_frames, spectrogram_begin, total_spectrogram_frames)
                else:
                    yield (None, None, spectrogram_begin, total_spectrogram_frames)
            else:
                yield (begin, total_clip_frames, None, None)
'''

'''
Callable that provides the bounds to extract sub-windows from an audio clip and/or spectrogram
to repeat the random window extraction from the paper https://arxiv.org/pdf/1608.04363v2.pdf
'''
class SingleWindowSelector:
    '''
    Args:
        - window_size_seconds
    OPTIONAL:
        - ...
        - random_location: determines wether we extract the clip sub-window from a random beginning index
    '''
    def __init__(self, window_size_seconds, sampling_rate = 22050, spectrogram_hop_length=None, random_location = True):
        self.window_size_seconds = window_size_seconds
        self.sampling_rate = sampling_rate

        self.window_size = int(math.floor(window_size_seconds * sampling_rate))

        self.spectrogram_hop_length = spectrogram_hop_length
        if self.spectrogram_hop_length is not None:
            self.spectrogram_window_size = int(math.ceil(self.window_size/spectrogram_hop_length))

        self.random_location = random_location

    '''
    When called, this instance will return a tuple of 4 values (AUDIO_WINDOW_BEGIN, AUDIO_WINDOW_END, SPECTROGRAM_WINDOW_BEGIN, SPECTROGRAM_WINDOW_END)
    providing the bounds for the selected sub-window on the audio clip and/or spectrogram
    Args:
    OPTIONAL
        - total_clip_frames: if provided, the returned tuple will contain the sub-windows bounds on the audio clip
        - total_spectrogram_frames: if provided, the returned tuple will contain the sub-windows bounds on the spectrogram
    '''
    def __call__(self, total_clip_frames=None, total_spectrogram_frames=None):
        assert total_clip_frames is not None or total_spectrogram_frames is not None, "To generate segments at least one among the clip length or the spectrogram length should be provided"
        if total_clip_frames is not None:
            assert total_clip_frames >= self.window_size, "Window size ({}) is bigger than audio clip length ({})".format(self.window_size, total_clip_frames)
        
        if total_clip_frames is not None:
            if self.random_location:
                begin = int(np.random.randint(0, total_clip_frames-self.window_size))
            else:
                begin = 0

        if total_spectrogram_frames is not None:
            assert self.spectrogram_hop_length is not None, "Specify a hop length for the spectrogram"

            if total_clip_frames is None:            
                if self.random_location:
                    spectrogram_begin = int(np.random.randint(0, total_spectrogram_frames-self.spectrogram_window_size))
                else:
                    spectrogram_begin = 0
            else:
                spectrogram_begin = int(math.floor(begin/self.spectrogram_hop_length))

            if total_clip_frames is not None:
                yield (begin, begin+self.window_size, spectrogram_begin, spectrogram_begin+self.spectrogram_window_size)
            else:
                yield (None, None, spectrogram_begin, spectrogram_begin+self.spectrogram_window_size)
        else:
            yield (begin, begin+self.window_size, None, None)
