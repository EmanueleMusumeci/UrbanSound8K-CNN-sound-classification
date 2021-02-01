import os
import sys
import argparse

import torch

from Dataset import SoundDatasetFold
from DataLoader import DataLoader
from nn.convolutional_model import CustomConvolutionalNetwork
from nn.paper_convolutional_model import PaperConvolutionalNetwork
from nn.feed_forward_model import FeedForwardNetwork
from data_augmentation.image_transformations import *
from data_augmentation.audio_transformations import *
from Trainer import *
from utils.dataset_utils import *
from utils.audio_utils import *


parser = argparse.ArgumentParser(description='Launch training with pre-compacted folds (use compact_dataset to \
                                              prepare them).')
parser.add_argument('-n','--name', type=str, help='Name of the training instance (will be the same name of the folder where \
                                        checkpoints are saved)')
#parser.add_argument('--preprocessing_name', help='Name of the preprocessing applied (pre-processed compacted folds \
#                                                        need to be generated with compact_dataset.py')
#parser.add_argument('--device', help='Name of the CUDA device to be used')


args = parser.parse_args()

#CUDA device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Directories
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
DATASET_DIR = os.path.join(BASE_DIR,"data")
MODEL_DIR = os.path.join(BASE_DIR,"model")

#Preprocessing
preprocessing_name = None
#preprocessing_name = "PitchShift"
#preprocessing_name = "TimeStretch"
#preprocessing_name = "DynamicRangeCompression"
#preprocessing_name = "BackgroundNoise"

CLIP_SECONDS = 3
SPECTROGRAM_HOP_LENGTH = 512
SAMPLE_RATE = 22050

COMPUTE_DELTAS = False
COMPUTE_DELTA_DELTAS = False
APPLY_IMAGE_AUGMENTATIONS = False

BATCH_SIZE = 128

#Spectrogram shape
spectrogram_frames_per_segment = CLIP_SECONDS*SAMPLE_RATE / SPECTROGRAM_HOP_LENGTH
spectrogram_bands = 128

in_channels = 1
if COMPUTE_DELTAS:in_channels = 2
elif COMPUTE_DELTA_DELTAS: in_channels = 3


#Dataset
DATASET_NAME = "UrbanSound8K"
DATASET_PERCENTAGE = 1.0
DEBUG_PREPROCESSING = False
DEBUG_TIMING = False

selected_classes = [0,1,2,3,4,5,6,7,8,9]

#Precompacted folds used for training
SINGLE_FOLD = False
if SINGLE_FOLD:
    train_fold_list = [1]
    test_fold_list = [1]
else:
    train_fold_list = [1,2,3,4,5,6,7,8,9]
    test_fold_list = [10]


#Model
USE_PAPER_CNN = True
CNN_INPUT_SIZE = (spectrogram_bands, spectrogram_frames_per_segment, in_channels)
FFN_INPUT_SIZE = 154


#Training instance name
if args.name is None:
    INSTANCE_NAME = (preprocessing_name if preprocessing_name is not None else "Base")
else:
    INSTANCE_NAME = args.name
if not USE_PAPER_CNN:
    INSTANCE_NAME+="_custom"


#Check we are not overwriting any existing checkpoints
PREVENT_OVERWRITE = True
if PREVENT_OVERWRITE:
    assert not os.path.exists(os.path.join(MODEL_DIR,INSTANCE_NAME)), "ATTENTION! The folder {} already exists: rename it or remove it".format(os.path.join(MODEL_DIR,INSTANCE_NAME))


#Image augmentations
if APPLY_IMAGE_AUGMENTATIONS:
    right_shift_transformation = SpectrogramShift(input_size=CNN_INPUT_SIZE,width_shift_range=4,shift_prob=0.9)
    left_shift_transformation = SpectrogramShift(input_size=CNN_INPUT_SIZE,width_shift_range=4,shift_prob=0.9, left=True)
    random_side_shift_transformation = SpectrogramShift(input_size=CNN_INPUT_SIZE,width_shift_range=4,shift_prob=0.9, random_side=True)
    background_noise_transformation = SpectrogramAddGaussNoise(input_size=CNN_INPUT_SIZE,prob_to_have_noise=0.55)
else:
    right_shift_transformation = None
    left_shift_transformation = None
    random_side_shift_transformation = None
    background_noise_transformation = None


#Load precompacted dataset
#Load preprocessed folds
if preprocessing_name is not None:
    train_audio_meta, train_audio_clips, train_audio_spectrograms = load_preprocessed_compacted_dataset(DATASET_DIR, preprocessing_name, folds = train_fold_list, only_spectrograms=True)
    #_, _, raw_train_audio_spectrograms = load_raw_compacted_dataset(DATASET_DIR, folds = train_fold_list)
else:
    train_audio_meta, train_audio_clips, train_audio_spectrograms = load_raw_compacted_dataset(DATASET_DIR, folds = train_fold_list, only_spectrograms=True)

#Load raw folds
test_audio_meta, test_audio_clips, test_audio_spectrograms = load_raw_compacted_dataset(DATASET_DIR, folds = test_fold_list, only_spectrograms=True)
#Free up memory
del train_audio_clips
del test_audio_clips


#Dataset instances
train_dataset = SoundDatasetFold(DATASET_DIR, DATASET_NAME, 
                            folds = train_fold_list, 
                            preprocessing_name = preprocessing_name,
                            audio_meta = train_audio_meta,
                            audio_clips = None,
                            audio_spectrograms = train_audio_spectrograms,
                            shuffle = True, 
                            use_spectrograms = True, 
                            image_shift_transformation = right_shift_transformation, 
                            image_background_noise_transformation = background_noise_transformation, 

                            spectrogram_frames_per_segment = spectrogram_frames_per_segment, 
                            spectrogram_bands = spectrogram_bands, 
                            compute_deltas=COMPUTE_DELTAS, 
                            compute_delta_deltas=COMPUTE_DELTA_DELTAS, 
                            test = False, 
                            progress_bar = True,
                            selected_classes=selected_classes,
                            select_percentage_of_dataset=DATASET_PERCENTAGE,
                            audio_segment_selector=SingleWindowSelector(CLIP_SECONDS, spectrogram_hop_length=SPECTROGRAM_HOP_LENGTH, random_location = True),
                            debug_preprocessing=DEBUG_PREPROCESSING,
                            debug_preprocessing_time=DEBUG_TIMING,
                            silent_clip_cutoff_dB = None
                            )   

test_dataset = SoundDatasetFold(DATASET_DIR, DATASET_NAME,  
                            folds = test_fold_list, 
                            preprocessing_name = preprocessing_name,
                            audio_meta = test_audio_meta,
                            audio_clips = None,
                            audio_spectrograms = test_audio_spectrograms,
                            shuffle = False, 
                            use_spectrograms = True, 
                            spectrogram_frames_per_segment = spectrogram_frames_per_segment, 
                            spectrogram_bands = spectrogram_bands, 
                            compute_deltas=False, 
                            compute_delta_deltas=False, 
                            test = True, 
                            progress_bar = True,
                            selected_classes=selected_classes,
                            select_percentage_of_dataset=DATASET_PERCENTAGE,
                            audio_segment_selector=SingleWindowSelector(CLIP_SECONDS, spectrogram_hop_length=SPECTROGRAM_HOP_LENGTH),
                            debug_preprocessing=DEBUG_PREPROCESSING,
                            debug_preprocessing_time=DEBUG_TIMING,
                            silent_clip_cutoff_dB = None
                            )

#Dataset statistics
num_classes = train_dataset.get_num_classes()
print("Number of classes: ", train_dataset.get_num_classes())
#print("Class names: ",train_dataset.class_distribution.keys())


#DataLoader instances
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


#Model instances
if USE_PAPER_CNN:
    model = PaperConvolutionalNetwork(CNN_INPUT_SIZE)
else:
    model = CustomConvolutionalNetwork(CNN_INPUT_SIZE)


#Loss function
loss_function = torch.nn.CrossEntropyLoss()


#Optimizer
if isinstance(model, PaperConvolutionalNetwork):
    optimizer = optim.SGD([
                {'params': model.convolutional_layers.parameters()},
                {'params': model.dense_layers.parameters(), 'weight_decay': 1e-3}
            ], lr=1e-2)
else:
    optimizer = torch.optim.Adam(model.parameters())


#Training loop wrapper
trainer = Trainer(
                    INSTANCE_NAME,
                    BATCH_SIZE,
                    train_loader,
                    test_loader,
                    model,
                    loss_function,
                    optimizer,
                    DEVICE,
                    MODEL_DIR,
                    lr_scheduler=None,
                    cnn = True
                )


#Launch training
trainer.train(50, save_test_scores_every=1, save_train_scores_every=1, save_model_every=1, compute_gradient_statistics=True)
