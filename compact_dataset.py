
import os
import argparse
import shutils

from utils.timing import function_timer, code_timer
from utils.dataset_utils import *
from utils.spectrogram_utils import display_heatmap, generate_mel_spectrogram_librosa, compute_spectrogram_frames
from utils.audio_utils import play_sound
from data_augmentation.audio_transformations import *

parser = argparse.ArgumentParser(description=\
    'Compact and, if required, preprocess the dataset, saving the result in memory-mapped folds. All clips are \
     cut/padded to a length of 4 seconds (that is the MEAN duration of the UrbanSound8K dataset) and this setting \
     is not changeable (there is not point to do so!). The default sampling rate is 22.05 kHz. Clips can be resampled \
     to another sampling rate using the --resample_to command-line argument. \
     Spectrograms are log-scale by default and have a default hop length of 512, default number of frequency bands of 128 \
     and the default sample rate of 22.05 kHz (these can all be tuned as command-line arguments). \
     NOTICE!!! If the sampling rate or percentage of a fold is change, use --overwrite_existing_folds command-line argument to \
     regenerate folds with the new parameters.')

parser.add_argument('--preprocessing_name', 
                    type=str, choices = ["PitchShift1", "PitchShift2", "TimeStretch", "DynamicRangeCompression", "BackgroundNoise"],
                    help='Name of the preprocessing applied while compacting')

parser.add_argument('--dataset_dir', 
                    type=str,
                    help='FULL path to the directory containing the UrbanSound8K dataset. In this directory also all the compacted folds will be saved. \
                    If not specified, the dataset is expected to be found in the "data" subdir of this repository root dir')

parser.add_argument('--overwrite_existing_folds', 
                    default=False, action='store_true',
                    help='Overwrites existing folds, to be used')

parser.add_argument('--apply_all_preprocessing', 
                    default=False, action='store_true',
                    help='Apply all preprocessings during compacting')

parser.add_argument('--normalize_audio_clips', 
                    default=False, action='store_true',
                    help='PREPROCESSING. Normalizes audio clips volume during compacting \
                    (NOTICE: ONLY AFFECTS THE RAW FOLDS so if raw folds have already been generated the \
                    --overwrite_existing_folds command-line argument should be used to re-generate them)')

parser.add_argument('--linear_scale', 
                    default=False, action='store_true',
                    help='PREPROCESSING. Compute spectrograms in the log-scale')

parser.add_argument('--resample_to', 
                    type=int, default = 22050,
                    help='PREPROCESSING. Resamples clips to a certain sampling rate (22050 (Hz) by default) \
                    (NOTICE: ONLY AFFECTS THE RAW FOLDS so if raw folds have already been generated the \
                    --overwrite_existing_folds command-line argument should be used to re-generate them, \
                     otherwise spectrograms might be incorrect (using a different sample rate))')

parser.add_argument('--spectrogram_hop_length', 
                    type=int, default = 512,
                    help='PREPROCESSING. Change spectrogram hop length (512 frames by default)')

parser.add_argument('--spectrogram_bands', 
                    type=int, default = 128,
                    help='PREPROCESSING. Change spectrogram bands (128 bands by default)')

parser.add_argument('--fold_percentage', 
                    type=float, default = 1.0,
                    help='Percentage of samples of each fold to be pre-processed and compacted (should be in range (0.0, 1.0] )')
                    
args = parser.parse_args()

base_dir = os.path.dirname(os.path.realpath(__file__))
if args.dataset_dir is None:
    DATASET_DIR = os.path.join(base_dir,"data")
else:
    DATASET_DIR = args.dataset_dir

fold_list = [1,2,3,4,5,6,7,8,9,10]

if args.fold_percentage is not None:
    assert args.fold_percentage>0.0 and args.fold_percentage<=1.0, "--fold_percentage should be in range (0.0, 1.0]"
    FOLD_PERCENTAGE = args.fold_percentage
else:
    FOLD_PERCENTAGE = 1.0

if args.overwrite_existing_folds:
    REGENERATE_FOLD = [True for fold in fold_list]

if args.apply_all_preprocessing:
    audio_augmentations = [
                        PitchShift(values = [-2, -1, 1, 2], name="PitchShift1"),
                        PitchShift(values = [-3.5, -2.5, 2.5, 3.5], name="PitchShift2"),
                        MUDADynamicRangeCompression(),
                        BackgroundNoise({
                            "street_scene_1" : "150993__saphe__street-scene-1.wav",
                            "street_scene_3" : "173955__saphe__street-scene-3.wav",
                            "street_valencia" : "207208__jormarp__high-street-of-gandia-valencia-spain.wav",
                            "city_park_tel_aviv" : "268903__yonts__city-park-tel-aviv-israel.wav",
                        }, files_dir = os.path.join(DATASET_DIR, "UrbanSound8K-JAMS", "background_noise")),
                        TimeStretch(values = [0.81, 0.93, 1.07, 1.23])
                        ]
else:
    if args.preprocessing_name == "PitchShift1":
        audio_augmentations = [PitchShift(values = [-2, -1, 1, 2], name="PitchShift1")]
    elif args.preprocessing_name == "PitchShift2":
        audio_augmentations = [PitchShift(values = [-3.5, -2.5, 2.5, 3.5], name="PitchShift2")]
    elif args.preprocessing_name == "TimeStretch":
        audio_augmentations = [TimeStretch(values = [0.81, 0.93, 1.07, 1.23], name="TimeStretch")]
    elif args.preprocessing_name == "DynamicRangeCompression":
        audio_augmentations = [MUDADynamicRangeCompression()]
    elif args.preprocessing_name == "BackgroundNoise":
        audio_augmentations = [BackgroundNoise({
                                "street_scene_1" : "150993__saphe__street-scene-1.wav",
                                "street_scene_3" : "173955__saphe__street-scene-3.wav",
                                "street_valencia" : "207208__jormarp__high-street-of-gandia-valencia-spain.wav",
                                "city_park_tel_aviv" : "268903__yonts__city-park-tel-aviv-israel.wav",
                                }, files_dir = os.path.join(DATASET_DIR, "UrbanSound8K-JAMS", "background_noise"))]
    elif args.preprocessing_name is None:
        audio_augmentations = []

if len(audio_augmentations)>0:
    for audio_augmentation in audio_augmentations:
        print("PREPROCESSING NAME: {}".format(audio_augmentation.name))
        for fold in fold_list:
            if not REGENERATE_FOLD[fold-1] and not check_audio_length(fold, DATASET_DIR, sampling_rate=args.resample_to):
                print("The compacted raw fold has a different audio length (which might be due to a different sample rate). Restart the compacting with the --overwrite_existing_folds command-line argument to correctly re-generate the raw fold.")
                exit()
            if not os.path.exists(os.path.join(DATASET_DIR, "raw", "urban_audio_fold_"+str(fold)+".dat")) or REGENERATE_FOLD[fold-1]:
                compact_raw_fold(DATASET_DIR, fold, normalize_audio=args.normalize_audio_clips, fold_percentage=FOLD_PERCENTAGE, resample_to=args.resample_to)
                REGENERATE_FOLD[fold-1] = False
            generate_compacted_fold_spectrograms(DATASET_DIR, fold, 
                                                sample_rate=args.resample_to, spectrogram_bands=args.spectrogram_bands, 
                                                hop_length=args.spectrogram_hop_length, log_mel=not args.linear_scale)

            generate_compacted_preprocessed_fold(DATASET_DIR, fold, audio_augmentation.name, audio_augmentation)
            generate_compacted_fold_spectrograms(DATASET_DIR, fold, preprocessing_name=audio_augmentation.name, 
                                                sample_rate=args.resample_to, spectrogram_bands=args.spectrogram_bands, 
                                                hop_length=args.spectrogram_hop_length, log_mel=not args.linear_scale)
        
        with code_timer("load_compacted_dataset", debug=True):
            audio_meta, audio_raw, audio_spectrograms_raw = load_raw_compacted_dataset(DATASET_DIR, folds = fold_list)

        with code_timer("load_compacted_preprocessed_dataset", debug=True):
            audio_meta, audio_preprocessed, audio_spectrograms_preprocessed = load_preprocessed_compacted_dataset(DATASET_DIR, preprocessing_name=audio_augmentation.name, folds = fold_list)
else:
    for fold in fold_list:
        if not REGENERATE_FOLD[fold-1] and not check_audio_length(fold, DATASET_DIR, sampling_rate=args.resample_to):
            print("The compacted raw fold has a different audio length (which might be due to a different sample rate). Restart the compacting with the --overwrite_existing_folds command-line argument to correctly re-generate the raw fold.")
            exit()
        if not os.path.exists(os.path.join(DATASET_DIR, "raw", "urban_audio_fold_"+str(fold)+".dat")) or REGENERATE_FOLD[fold-1]:
            compact_raw_fold(DATASET_DIR, fold, normalize_audio=args.normalize_audio_clips, fold_percentage=FOLD_PERCENTAGE, resample_to=args.resample_to)
            REGENERATE_FOLD[fold-1] = False
        generate_compacted_fold_spectrograms(DATASET_DIR, fold, 
                                                sample_rate=args.resample_to, spectrogram_bands=args.spectrogram_bands, 
                                                hop_length=args.spectrogram_hop_length, log_mel=not args.linear_scale)
        
        with code_timer("load_compacted_dataset", debug=True):
            audio_meta, audio_raw, audio_spectrograms_raw = load_raw_compacted_dataset(DATASET_DIR, folds = fold_list)

        with code_timer("load_compacted_preprocessed_dataset", debug=True):
            audio_meta, audio_preprocessed, audio_spectrograms_preprocessed = load_preprocessed_compacted_dataset(DATASET_DIR, preprocessing_name=audio_augmentation.name, folds = fold_list)
'''
for i, clip_raw in enumerate(audio_raw):
    print("{}: {} ({})".format(i, clip_raw, len(audio_raw)))
    play_sound(clip_raw)
    if audio_spectrograms_raw is not None:
        print(audio_spectrograms_raw.shape)
        display_heatmap(audio_spectrograms_raw[i])

    for preprocessing_value, preprocessed_clips in audio_preprocessed.items():
        print("{}: {} ({}) value: {}".format(i, preprocessed_clips[i], len(preprocessed_clips[i]), preprocessing_value))
        if audio_spectrograms_preprocessed is not None:
            display_heatmap(audio_spectrograms_preprocessed[preprocessing_value][i])
        play_sound(preprocessed_clips[i])
    break
'''