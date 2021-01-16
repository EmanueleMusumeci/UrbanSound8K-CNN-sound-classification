import os

from Dataset import SoundDatasetFold
from DataLoader import DataLoader
from convolutional_model import ConvolutionalNetwork
from image_transformations import *


base_dir = os.path.dirname(os.path.realpath(__file__))
DATASET_DIR = os.path.join(base_dir,"data")
DATASET_NAME = "UrbanSound8K"

use_CNN = True

spectrogram_frames_per_segment = 41
spectrogram_bands = 60
in_channels = (3 if use_CNN else 1)

CNN_INPUT_SIZE = (spectrogram_bands, spectrogram_frames_per_segment, in_channels)

right_shift_transformation = SpectrogramShift(input_size=CNN_INPUT_SIZE,width_shift_range=4,shift_prob=0.9)
left_shift_transformation = SpectrogramShift(input_size=CNN_INPUT_SIZE,width_shift_range=4,shift_prob=0.9, left=True)
random_side_shift_transformation = SpectrogramShift(input_size=CNN_INPUT_SIZE,width_shift_range=4,shift_prob=0.9, random_side=True)

background_noise_transformation = SpectrogramAddGaussNoise(input_size=CNN_INPUT_SIZE,prob_to_have_noise=0.55)

dataset = SoundDatasetFold(DATASET_DIR, DATASET_NAME, 
                        folds = [1], shuffle_dataset = True, 
                        generate_spectrograms = False, 
                        shift_transformation = right_shift_transformation, 
                        background_noise_transformation = background_noise_transformation, 
                        audio_augmentation_pipeline = [], 
                        spectrogram_frames_per_segment = spectrogram_frames_per_segment, 
                        spectrogram_bands = spectrogram_bands, 
                        compute_deltas=True, 
                        compute_delta_deltas=True, 
                        test = False, 
                        progress_bar = True
                        )

dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

net = ConvolutionalNetwork(CNN_INPUT_SIZE)

batch = next(iter(dataloader))

print(net(batch["preprocessed_spectrograms"]))