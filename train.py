import os

import torch

from Dataset import SoundDatasetFold
from DataLoader import DataLoader
from nn.convolutional_model import ConvolutionalNetwork
from nn.paper_convolutional_model import PaperConvolutionalNetwork
from nn.feed_forward_model import FeedForwardNetwork
from data_augmentation.image_transformations import *
from data_augmentation.audio_transformations import *
from Trainer import *
from utils.dataset_utils import *
from utils.audio_utils import *

preprocessing_name = None
#preprocessing_name = "PitchShift"


CUSTOM_MODEL = False

INSTANCE_NAME = (preprocessing_name if preprocessing_name is not None else "Base")
 
if CUSTOM_MODEL:
    INSTANCE_NAME+="_custom"

BATCH_SIZE = 128

USE_CNN = True


APPLY_IMAGE_AUGMENTATIONS = False
#APPLY_AUDIO_AUGMENTATIONS = True
CLIP_SECONDS = 3
SPECTROGRAM_HOP_LENGTH = 512
SAMPLE_RATE = 22050

COMPUTE_DELTAS = False
COMPUTE_DELTA_DELTAS = False

DEBUG_PREPROCESSING = False
DEBUG_TIMING = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
  from google.colab import drive
  base_dir = "/content/drive/My Drive/Neural Networks Project"
  DATASET_DIR = "/content/drive/data"
except:
  base_dir = os.path.dirname(os.path.realpath(__file__))
  DATASET_DIR = os.path.join(base_dir,"data")

DATASET_NAME = "UrbanSound8K"
DATASET_PERCENTAGE = 1.0

MODEL_DIR = os.path.join(base_dir,"model")


assert not os.path.exists(os.path.join(MODEL_DIR,INSTANCE_NAME)), "ATTENTION! The folder {} already exists: rename it or remove it".format(os.path.join(MODEL_DIR,INSTANCE_NAME))


selected_classes = [0,1,2,3,4,5,6,7,8,9]

spectrogram_frames_per_segment = CLIP_SECONDS*SAMPLE_RATE / SPECTROGRAM_HOP_LENGTH
spectrogram_bands = 128

in_channels = 1
if USE_CNN:
    if COMPUTE_DELTAS:in_channels = 2
    elif COMPUTE_DELTA_DELTAS: in_channels = 3

CNN_INPUT_SIZE = (spectrogram_bands, spectrogram_frames_per_segment, in_channels)
FFN_INPUT_SIZE = 154

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

#Audio augmentations
#if APPLY_AUDIO_AUGMENTATIONS:
#    random_pitch_shift = PitchShift([-3.5, -2.5, 2.5, 3.5], debug_time=DEBUG_TIMING)
#    random_time_stretch = TimeStretch([0.81, 0.93, 1.07, 1.23], debug_time=DEBUG_TIMING)
#else:
#    random_pitch_shift = None
#    random_time_stretch = None


train_fold_list = [1]
train_fold_list = [1,2,3,4,5,6,7,8,9]
test_fold_list = [10]

if preprocessing_name is not None:
    train_audio_meta, train_audio_clips, train_audio_spectrograms = load_preprocessed_compacted_dataset(DATASET_DIR, preprocessing_name, folds = train_fold_list, only_spectrograms=True)
    #_, _, raw_train_audio_spectrograms = load_raw_compacted_dataset(DATASET_DIR, folds = train_fold_list)
else:
    train_audio_meta, train_audio_clips, train_audio_spectrograms = load_raw_compacted_dataset(DATASET_DIR, folds = train_fold_list, only_spectrograms=True)

test_audio_meta, test_audio_clips, test_audio_spectrograms = load_raw_compacted_dataset(DATASET_DIR, folds = test_fold_list, only_spectrograms=True)

del train_audio_clips
del test_audio_clips

train_dataset = SoundDatasetFold(DATASET_DIR, DATASET_NAME, 
                            folds = train_fold_list, 
                            preprocessing_name = preprocessing_name,
                            audio_meta = train_audio_meta,
                            audio_clips = None,
                            audio_spectrograms = train_audio_spectrograms,
                            shuffle = True, 
                            use_spectrograms = USE_CNN, 
                            image_shift_transformation = right_shift_transformation, 
                            image_background_noise_transformation = background_noise_transformation, 
                            #time_stretch_transformation = random_time_stretch,
                            #pitch_shift_transformation = random_pitch_shift, 
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
                            use_spectrograms = USE_CNN, 
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

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


if USE_CNN:
    #model = ConvolutionalNetwork(CNN_INPUT_SIZE)
    model = PaperConvolutionalNetwork(CNN_INPUT_SIZE)
else:
    model = FeedForwardNetwork(FFN_INPUT_SIZE, 256, train_dataset.get_num_classes())

num_classes = train_dataset.get_num_classes()
print("Number of classes: ", train_dataset.get_num_classes())
print("Class names: ",train_dataset.class_distribution.keys())

loss_function = torch.nn.CrossEntropyLoss()

if isinstance(model, PaperConvolutionalNetwork):
    optimizer = optim.SGD([
                {'params': model.convolutional_layers.parameters()},
                {'params': model.dense_layers.parameters(), 'weight_decay': 1e-3}
            ], lr=1e-2)
else:
    optimizer = torch.optim.Adam(model.parameters())

trainer = Trainer(
                    INSTANCE_NAME,
                    BATCH_SIZE,
                    train_loader,
                    test_loader,
                    train_dataset.get_id_to_class(),
                    model,
                    loss_function,
                    optimizer,
                    DEVICE,
                    MODEL_DIR,
                    lr_scheduler=None,
                    cnn = USE_CNN
                )

trainer.train(50, save_test_scores_every=1, save_train_scores_every=1, save_model_every=1, compute_gradient_statistics=True)
