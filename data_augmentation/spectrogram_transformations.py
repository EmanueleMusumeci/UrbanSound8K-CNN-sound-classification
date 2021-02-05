
import random

import torch
import torchaudio
from torchaudio import transforms

from spec_augment_master import *
from spec_augment_master.exp.nb_SparseImageWarp import sparse_image_warp

from collections import namedtuple

import matplotlib.pyplot as plt
import os

def tensor_to_img(spectrogram):
    plt.figure(figsize=(14,1)) # arbitrary, looks good on my screen.
    plt.imshow(spectrogram[0])
    plt.show()
    display(spectrogram.shape)

def tfm_spectro(ad:Audio, sr=44100, to_db_scale=False, n_fft=1024, ws=None, hop=None, f_min=0.0, f_max=-80, pad=0, n_mels=128):
    # We must reshape signal for torchaudio to generate the spectrogram.
    mel = transforms.MelSpectrogram(sample_rate=ad.sr, n_mels=n_mels, n_fft=n_fft, win_length=ws, hop_length=hop, f_min=f_min, f_max=f_max, pad=pad,)(ad.sig.reshape(1, -1))
    mel = mel.permute(0,1,2) # fixed shape
    print(f_max)
    if to_db_scale: mel = torchaudio.transforms.AmplitudeToDB(stype='magnitude', top_db=f_max)(mel)#changed, last was deprecated
    return mel

def tensor_to_img(spectrogram):
        plt.figure(figsize=(14,1)) # arbitrary, looks good on my screen.
        plt.imshow(spectrogram[0])
        plt.show()
        display(spectrogram.shape)

class TimeMaskSpectrogram(object):
    def __init__(self):
        print("TimeMaskSpectrogram")

    def __call__(self,spec, T=40, num_masks=1, replace_with_zero=False):
        '''
        Implements the Time Mask transformation 
        Args:
          - spec: spectrogram on which apply the transformation
          OPTIONAL:
            - T: 
            - num_mask: number of mask to use
            - replace_with_zero:
        '''
        cloned = spec.clone()
        len_spectro = cloned.shape[2]
        
        for i in range(0, num_masks):
            t = random.randrange(0, T)
            t_zero = random.randrange(0, len_spectro - t)

            # avoids randrange error if values are equal and range is empty
            if (t_zero == t_zero + t): return cloned

            mask_end = random.randrange(t_zero, t_zero + t)
            if (replace_with_zero): cloned[0][:,t_zero:mask_end] = 0
            else: cloned[0][:,t_zero:mask_end] = cloned.mean()
        return cloned


class TimeWarpSpectrogram(object):
    def __init__(self):
        print("TimeWarpSpectrogram")

    def __call__(self,spec, W=5):
        '''
        Implements the Time Warping transformation 
        Args:
          - spec: spectrogram on which apply the transformation
          OPTIONAL:
            - W: warping parameter
        '''
        num_rows = spec.shape[1]
        spec_len = spec.shape[2]
        device = spec.device
        
        y = num_rows//2
        horizontal_line_at_ctr = spec[0][y]
        assert len(horizontal_line_at_ctr) == spec_len
        
        point_to_warp = horizontal_line_at_ctr[random.randrange(W, spec_len - W)]
        assert isinstance(point_to_warp, torch.Tensor)

        # Uniform distribution from (0,W) with chance to be up to W negative
        dist_to_warp = random.randrange(-W, W)
        src_pts, dest_pts = (torch.tensor([[[y, point_to_warp]]], device=device), 
                            torch.tensor([[[y, point_to_warp + dist_to_warp]]], device=device))
        warped_spectro, dense_flows = sparse_image_warp(spec, src_pts, dest_pts)
        return warped_spectro.squeeze(3)


class FrequencyMaskSpectrogram(object):
    def __init__(self):
        print("FrequencyMaskSpectrogram")
    def __call__(self,spec, F=30, num_masks=1, replace_with_zero=False):
        '''
        Implements the Frequency Mask transformation 
        Args:
          - spec: spectrogram on which apply the transformation
          OPTIONAL:
            - F: 
            - num_mask: number of mask to use
            - replace_with_zero:
        '''
        cloned = spec.clone()
        num_mel_channels = cloned.shape[1]
        
        for i in range(0, num_masks):        
            f = random.randrange(0, F)
            f_zero = random.randrange(0, num_mel_channels - f)

            # avoids randrange error if values are equal and range is empty
            if (f_zero == f_zero + f): return cloned

            mask_end = random.randrange(f_zero, f_zero + f) 
            if (replace_with_zero): cloned[0][f_zero:mask_end] = 0
            else: cloned[0][f_zero:mask_end] = cloned.mean()
        
        return cloned

