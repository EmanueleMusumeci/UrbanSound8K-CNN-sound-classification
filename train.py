import os
import sys
import argparse
import math

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


parser = argparse.ArgumentParser(description='Trains a neural network for the environmental sound classification task on the \
    UrbanSound8K dataset. Pre-processing mainly follows the paper https://arxiv.org/pdf/1608.04363v2.pdf for the sample sub-window \
    extraction. Some other pre-processing techniques were applied, leading to interesting results (see report.pdf for a in-depth analysis of our results). \
    Given the long time required to pre-process samples on-line, a dataset compacting utility is provided at compact_dataset.py (inspect the -h option of \
    compact_dataset.py for more info).')

#DATASET INITIALIZATION PARAMETERS
parser.add_argument('-n','--name', type=str, help='Name of the training instance (will be the same name of the folder where \
                                        checkpoints are saved)')

parser.add_argument('--device', 
                    type=str,
                    help='Name of the CUDA device to be used (using "cuda" will select the default cuda device). \
                        "cuda" is chosen by default if available else the cpu is selected.')

parser.add_argument('--dataset_dir', 
                    type=str,
                    help='FULL path to the directory containing the UrbanSound8K folder of the dataset. If not specified, the dataset is expected to be \
                    found in the "data" subdir of this repository root dir')

parser.add_argument('--model_dir', 
                    type=str,
                    help='FULL path to the directory where model checkpoints will be saved. If not specified, the dataset is expected to be \
                    found in the "model" subdir of this repository root dir')

parser.add_argument('--dataset_percentage', 
                    type=float, default = 1.0,
                    help='Percentage of the dataset to use for training (should be in range (0.0, 1.0] )')

parser.add_argument('--disable_model_overwrite_protection', 
                    default=False, action='store_true',
                    help='Disables the protection from accidental model checkpoint overwrites \
                        (when enabled, terminates the program if a model with the same name has already been trained)')

parser.add_argument('--load_compacted_audio', 
                    default=False, action='store_true',
                    help='If specified, audio clips are loaded from the dataset compacted folds, \
                          which should have been previously compacted using the command-line utility in compact_dataset.py. \
                          If a --preprocessing_name is specified, the compacted fold pre-processed with that audio preprocessing method \
                          will be loaded. \
                          (NOTICE: compacting the dataset before training requires from 30m up to 3h depending on the \
                          number of dataset folds that are compacted and the number of preprocessings applied, but greatly reduces training time). \
                          See the -h info from the compact_dataset.py utility.')

parser.add_argument('--load_compacted_spectrograms', 
                    default=False, action='store_true',
                    help='If specified, audio clips are loaded from the dataset compacted folds, \
                          which should have been previously compacted using the command-line utility in compact_dataset.py. \
                          If a --preprocessing_name is specified, the compacted fold pre-processed with that audio preprocessing method \
                          will be loaded. \
                          (NOTICE: compacting the dataset before training requires from 30m up to 3h depending on the \
                          number of dataset folds that are compacted and the number of preprocessings applied, but greatly reduces training time, \
                          up to a 13X speed-up factor when loading compacted spectrograms). See the -h info from the compact_dataset.py utility.  \
                          (NOTICE: when using pre-compacted spectrograms, the audio normalization step is disabled, as it would have to be performed \
                          on audio clips, anyway its impact is marginal to the training accuracy.)')

parser.add_argument('--test_mode', 
                    default=False, action='store_true',
                    help='Launches training with only one dataset fold and no preprocessing applied for a fast test of the on-line preprocessing.')

#PREPROCESSING PARAMETERS
parser.add_argument('--preprocessing_name', 
                    type=str, choices = ["PitchShift1", "PitchShift2", "TimeStretch", "DynamicRangeCompression", "BackgroundNoise"],
                    help='PREPROCESSING. Name of the AUDIO preprocessing method applied, from the ones discussed in report.pdf \n \
                        (NOTICE: pre-preprocessing samples while training is very slow: it might be a good choice to first pre-process and compact folds, \
                         saving them to memory-mapped files. In order to do so, follow the -h instructions from compact_dataset.py)')

parser.add_argument('--spectrogram_bands', 
                    type=int, default = 128,
                    help='PREPROCESSING. Spectrogram bands used during spectrogram generation \
                        (NOTICE: 1) only applied when on-line preprocessing is performed or when the --load_compacted_audio command-line option is used) \
                        2) changing the clip sub-window length will require changing the structure of the neural network as the current one only \
                          works with spectrograms of 128 bands, 512 hop length, sample rate of 22050, from 3 seconds audio clip)')

parser.add_argument('--clip_seconds', 
                    type=float, default = 3,
                    help='PREPROCESSING. Length of the sub-windows extracted from the clips during training (default 3 seconds) \
                         (NOTICE: changing the clip sub-window length will require changing the structure of the neural network as the current one only \
                          works with spectrograms of 128 bands, 512 hop length, sample rate of 22050, from 3 seconds audio clip)')

parser.add_argument('--normalize_audio_clips', 
                    default=False, action='store_true',
                    help='PREPROCESSING. Normalizes audio clips volume during training \
                        (NOTICE: only applied when on-line preprocessing is performed or when the --load_compacted_audio command-line option is used)')

parser.add_argument('--drop_silent_clips', 
                    default=False, action='store_true',
                    help='PREPROCESSING. Drops clips whose mean volume is lower than -70dB')

parser.add_argument('--compute_deltas', 
                    default=False, action='store_true',
                    help='PREPROCESSING. Computes delta values for each spectrogram during training \
                        (NOTICE: if not specified, delta delta training is disabled)')

parser.add_argument('--compute_delta_deltas', 
                    default=False, action='store_true',
                    help='PREPROCESSING. Computes delta-delta values for each spectrogram during training \
                        (NOTICE: enables delta training as well)')

parser.add_argument('--apply_spectrogram_image_background_noise', 
                    default=False, action='store_true',
                    help='PREPROCESSING. Applies a random gaussian background noise to spectrogram images as augmentation \
                        (NOTICE: spectrogram image augmentations gave the best results so far)')

parser.add_argument('--apply_spectrogram_image_shift', 
                    default=False, action='store_true',
                    help='PREPROCESSING. Applies a random right shift to spectrogram images as augmentation \
                        (NOTICE: spectrogram image augmentations gave the best results so far)')

parser.add_argument('--print_preprocessing_debug', 
                    default=False, action='store_true',
                    help='Verbose version of the preprocessing, for debug purposes. \
                        (NOTICE: enabling this option replaces the print function with tqdm.write to be compatible with the tqdm progress bar)')

#MODEL PARAMETERS
parser.add_argument('--custom_cnn', 
                    default=False, action='store_true',
                    help='Trains with our custom CNN (see report.pdf, \
                        else uses the cnn from the paper https://arxiv.org/pdf/1608.04363v2.pdf by default)')

parser.add_argument('--batch_size', 
                    type=int, default = 128,
                    help='Training batch size')


args = parser.parse_args()

#CUDA device
if args.device is None:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    DEVICE = args.device

#Directories
BASE_DIR = os.path.dirname(os.path.realpath(__file__))

if args.dataset_dir is None:
    DATASET_DIR = os.path.join(BASE_DIR,"data")
else:
    DATASET_DIR = args.dataset_dir

if args.model_dir is None:
    MODEL_DIR = os.path.join(BASE_DIR,"model")
else:
    MODEL_DIR = args.model_dir

#Preprocessing applied
if args.preprocessing_name is not None:
    preprocessing_name = args.preprocessing_name
else:
    preprocessing_name = None

#These are left commented to manually control the preprocessing applied   
#preprocessing_name = None
#preprocessing_name = "PitchShift1"
#preprocessing_name = "PitchShift2"
#preprocessing_name = "TimeStretch"
#preprocessing_name = "DynamicRangeCompression"
#preprocessing_name = "BackgroundNoise"

if args.clip_seconds is None:
    CLIP_SECONDS = 3
else:
    CLIP_SECONDS = args.clip_seconds
    
SPECTROGRAM_HOP_LENGTH = 512
SAMPLE_RATE = 22050

#These parameters control whether we also compute Deltas and Delta-Deltas on spectrograms 
COMPUTE_DELTAS = args.compute_deltas
COMPUTE_DELTA_DELTAS = args.compute_delta_deltas
#COMPUTE_DELTAS = False
#COMPUTE_DELTA_DELTAS = False

if COMPUTE_DELTA_DELTAS:
    COMPUTE_DELTAS = True

#These parameters control whether we apply image augmentation techniques directly on the spectrograms
APPLY_IMAGE_SHIFT = args.apply_spectrogram_image_shift
APPLY_IMAGE_NOISE = args.apply_spectrogram_image_background_noise


#Spectrogram shape
spectrogram_frames_per_segment = math.ceil(CLIP_SECONDS*SAMPLE_RATE / SPECTROGRAM_HOP_LENGTH)
spectrogram_bands = args.spectrogram_bands


#Input shape of the CNN
in_channels = 1
if COMPUTE_DELTAS:in_channels = 2
if COMPUTE_DELTA_DELTAS: in_channels = 3


#Dataset
DATASET_NAME = "UrbanSound8K"

if args.dataset_percentage is not None:
    assert args.dataset_percentage>0.0 and args.dataset_percentage<=1.0, "--dataset_percentage should be in range (0.0, 1.0]"
    DATASET_PERCENTAGE = args.dataset_percentage
else:
    DATASET_PERCENTAGE = 1.0

DEBUG_PREPROCESSING = args.print_preprocessing_debug

DEBUG_TIMING = False

selected_classes = [0,1,2,3,4,5,6,7,8,9]

BATCH_SIZE = args.batch_size

#Precompacted folds to be used for training
if args.test_mode:
    SINGLE_FOLD = True
    APPLY_IMAGE_NOISE = False
    APPLY_IMAGE_SHIFT = False
    COMPUTE_DELTAS = False
    COMPUTE_DELTA_DELTAS = False
    preprocessing_name = None
else:
    SINGLE_FOLD = False

if SINGLE_FOLD:
    train_fold_list = [1]
    test_fold_list = [10]
else:
    train_fold_list = [1,2,3,4,5,6,7,8,9]
    test_fold_list = [10]


#Model
USE_PAPER_CNN = not args.custom_cnn
CNN_INPUT_SIZE = (spectrogram_bands, spectrogram_frames_per_segment, in_channels)
FFN_INPUT_SIZE = 154


#Training instance name
if args.name is None:
    INSTANCE_NAME = (preprocessing_name if preprocessing_name is not None else "Base")
else:
    INSTANCE_NAME = args.name
if not USE_PAPER_CNN:
    INSTANCE_NAME+="_custom"
if COMPUTE_DELTAS:
    INSTANCE_NAME+="_delta"
if COMPUTE_DELTA_DELTAS:
    INSTANCE_NAME+="_delta"

#Check we are not overwriting any existing checkpoints
PREVENT_OVERWRITE = not args.disable_model_overwrite_protection
if PREVENT_OVERWRITE:
    if(os.path.exists(os.path.join(MODEL_DIR,INSTANCE_NAME))):
        print("ATTENTION! The folder {} already exists: use the --disable_model_overwrite_protection command-line argument or rename/remove the folder".format(os.path.join(MODEL_DIR,INSTANCE_NAME)))
        exit()


#Image augmentations
if APPLY_IMAGE_SHIFT:
    shift_transformation = SpectrogramShift(input_size=CNN_INPUT_SIZE,width_shift_range=4,shift_prob=0.9)
    #left_shift_transformation = SpectrogramShift(input_size=CNN_INPUT_SIZE,width_shift_range=4,shift_prob=0.9, left=True)
    #random_side_shift_transformation = SpectrogramShift(input_size=CNN_INPUT_SIZE,width_shift_range=4,shift_prob=0.9, random_side=True)
else:
    shift_transformation = None
    #left_shift_transformation = None
    #random_side_shift_transformation = None
    
if APPLY_IMAGE_NOISE:
    background_noise_transformation = SpectrogramAddGaussNoise(input_size=CNN_INPUT_SIZE,prob_to_have_noise=0.55)
else:
    background_noise_transformation = None

NORMALIZE_AUDIO = args.normalize_audio_clips
if args.drop_silent_clips:
    silent_clip_cutoff_dB = -70
else:
    silent_clip_cutoff_dB = None
    
#Load precompacted dataset

preprocessor = None
LOAD_COMPACTED_AUDIO = args.load_compacted_audio
LOAD_COMPACTED_SPECTROGRAMS = args.load_compacted_spectrograms

if LOAD_COMPACTED_AUDIO and LOAD_COMPACTED_SPECTROGRAMS:
    print("Only one option among --load_compacted_audio and --load_compacted_spectrograms can be specified at the same time.")
    exit()

if LOAD_COMPACTED_AUDIO:
    #Load preprocessed folds
    if preprocessing_name is not None:
        train_audio_meta, train_audio_clips, train_audio_spectrograms = load_preprocessed_compacted_dataset(DATASET_DIR, preprocessing_name, folds = train_fold_list, only_spectrograms=False)
        #_, _, raw_train_audio_spectrograms = load_raw_compacted_dataset(DATASET_DIR, folds = train_fold_list)
    else:
        train_audio_meta, train_audio_clips, train_audio_spectrograms = load_raw_compacted_dataset(DATASET_DIR, folds = train_fold_list, only_spectrograms=False)

    #Load raw folds
    test_audio_meta, test_audio_clips, test_audio_spectrograms = load_raw_compacted_dataset(DATASET_DIR, folds = test_fold_list, only_spectrograms=False)
    
    #Free up memory
    del train_audio_spectrograms
    del test_audio_spectrograms
    train_audio_spectrograms = None
    test_audio_spectrograms = None

elif LOAD_COMPACTED_SPECTROGRAMS:
        #Load preprocessed folds
    if preprocessing_name is not None:
        train_audio_meta, train_audio_clips, train_audio_spectrograms = load_preprocessed_compacted_dataset(DATASET_DIR, preprocessing_name, folds = train_fold_list, only_spectrograms=True, 
                                                                                                            check_spectrogram_info = {"bands" : spectrogram_bands, "hop_length" : SPECTROGRAM_HOP_LENGTH, "sample_rate" : SAMPLE_RATE})
        #_, _, raw_train_audio_spectrograms = load_raw_compacted_dataset(DATASET_DIR, folds = train_fold_list)
    else:
        train_audio_meta, train_audio_clips, train_audio_spectrograms = load_raw_compacted_dataset(DATASET_DIR, folds = train_fold_list, only_spectrograms=True, 
                                                                                                            check_spectrogram_info = {"bands" : spectrogram_bands, "hop_length" : SPECTROGRAM_HOP_LENGTH, "sample_rate" : SAMPLE_RATE})

    #Load raw folds
    test_audio_meta, test_audio_clips, test_audio_spectrograms = load_raw_compacted_dataset(DATASET_DIR, folds = test_fold_list, only_spectrograms=True, 
                                                                                                            check_spectrogram_info = {"bands" : spectrogram_bands, "hop_length" : SPECTROGRAM_HOP_LENGTH, "sample_rate" : SAMPLE_RATE})
    
    #Free up memory
    del train_audio_clips
    del test_audio_clips
    train_audio_clips = None
    test_audio_clips = None

else:
    train_audio_meta = None
    train_audio_clips = None
    train_audio_spectrograms = None

    test_audio_meta = None
    test_audio_clips = None
    test_audio_spectrograms = None
                                
    if preprocessing_name == "PitchShift1":
        preprocessor = PitchShift(values = [-2, -1, 1, 2], name="PitchShift1")

    elif preprocessing_name == "PitchShift2":
        preprocessor = PitchShift(values = [-3.5, -2.5, 2.5, 3.5], name="PitchShift2")
    
    elif preprocessing_name == "TimeStretch":
        preprocessor = TimeStretch(values = [0.81, 0.93, 1.07, 1.23])

    elif preprocessing_name == "DynamicRangeCompression":
        preprocessor = MUDADynamicRangeCompression()

    elif preprocessing_name == "BackgroundNoise":
        preprocessor = BackgroundNoise({
                                        "street_scene_1" : "150993__saphe__street-scene-1.wav",
                                        "street_scene_3" : "173955__saphe__street-scene-3.wav",
                                        "street_valencia" : "207208__jormarp__high-street-of-gandia-valencia-spain.wav",
                                        "city_park_tel_aviv" : "268903__yonts__city-park-tel-aviv-israel.wav",
                                    }, files_dir = os.path.join(DATASET_DIR, "UrbanSound8K-JAMS", "background_noise"))

#Dataset instances
train_dataset = SoundDatasetFold(DATASET_DIR, DATASET_NAME, 
                            folds = train_fold_list, 
                            preprocessing_name = preprocessing_name,
                            preprocessor=preprocessor,
                            audio_meta = train_audio_meta,
                            audio_clips = None,
                            audio_spectrograms = train_audio_spectrograms,
                            shuffle = True, 
                            use_spectrograms = True, 
                            image_shift_transformation = shift_transformation, 
                            image_background_noise_transformation = background_noise_transformation, 

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
                            silent_clip_cutoff_dB = silent_clip_cutoff_dB,
                            normalize_audio=NORMALIZE_AUDIO
                            )   

test_dataset = SoundDatasetFold(DATASET_DIR, DATASET_NAME,  
                            folds = test_fold_list, 
                            preprocessing_name = None,
                            audio_meta = test_audio_meta,
                            audio_clips = None,
                            audio_spectrograms = test_audio_spectrograms,
                            shuffle = False, 
                            use_spectrograms = True, 

                            spectrogram_bands = spectrogram_bands, 
                            compute_deltas=COMPUTE_DELTAS, 
                            compute_delta_deltas=COMPUTE_DELTA_DELTAS, 
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


