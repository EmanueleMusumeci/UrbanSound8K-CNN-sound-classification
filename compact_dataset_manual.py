
import os
from core.utils.timing import function_timer, code_timer
from core.utils.dataset_utils import *
from core.utils.spectrogram_utils import display_heatmap, generate_mel_spectrogram_librosa, compute_spectrogram_frames
from core.utils.audio_utils import play_sound
from core.data_augmentation.audio_transformations import *


"""
NOTICE: While compact_dataset.py offers a command-line interface to control the dataset compacting process,
this file allows manually controlling dataset compacting, by manually setting the parameters in the code
"""

base_dir = os.path.dirname(os.path.realpath(__file__))
DATASET_DIR = os.path.join(base_dir,"data")
print(DATASET_DIR)

#delete_compacted_dataset(DATASET_DIR)

fold_list = [1,2,3,4,5,6,7,8,9,10]
FOLD_PERCENTAGE = 1.0


OVERWRITE_EXISTING_FOLDS = True
REGENERATE_FOLD = [False for fold in range(0,10)]
if OVERWRITE_EXISTING_FOLDS:
    for fold in fold_list:
        REGENERATE_FOLD[fold-1] = True

audio_augmentations = [
                        #PitchShift(values = [-2, -1, 1, 2]),
                        #PitchShift(values = [-3.5, -2.5, 2.5, 3.5]),
                        #MUDADynamicRangeCompression(),
                        #BackgroundNoise({
                        #    "street_scene_1" : "150993__saphe__street-scene-1.wav",
                        #    "street_scene_3" : "173955__saphe__street-scene-3.wav",
                        #    "street_valencia" : "207208__jormarp__high-street-of-gandia-valencia-spain.wav",
                        #    "city_park_tel_aviv" : "268903__yonts__city-park-tel-aviv-israel.wav",
                        #}, files_dir = os.path.join(DATASET_DIR, "UrbanSound8K-JAMS", "background_noise")),
                        #TimeStretch(values = [0.81, 0.93, 1.07, 1.23])
                        
                        ]

if len(audio_augmentations)>0:
    for audio_augmentation in audio_augmentations:
        print("PREPROCESSING NAME: {}".format(audio_augmentation.name))
        for fold in fold_list:
            if not REGENERATE_FOLD[fold-1] and not check_audio_length(fold, DATASET_DIR, sampling_rate=22050):
                print("The compacted raw fold has a different audio length (which might be due to a different sample rate). Restart the compacting with the --overwrite_existing_folds command-line argument to correctly re-generate the raw fold.")
                exit()
            if not os.path.exists(os.path.join(DATASET_DIR, "raw", "urban_audio_fold_"+str(fold)+".dat")) or REGENERATE_FOLD[fold-1]:
                compact_raw_fold(DATASET_DIR, fold, normalize_audio=False, fold_percentage=FOLD_PERCENTAGE, resample_to=22050)
                REGENERATE_FOLD[fold-1] = False
            generate_compacted_fold_spectrograms(DATASET_DIR, fold, 
                                                sample_rate=22050, spectrogram_bands=128, 
                                                hop_length=512, log_mel=True)

            generate_compacted_preprocessed_fold(DATASET_DIR, fold, audio_augmentation.name, audio_augmentation)
            generate_compacted_fold_spectrograms(DATASET_DIR, fold, preprocessing_name=audio_augmentation.name, 
                                                sample_rate=22050, spectrogram_bands=128, 
                                                hop_length=512, log_mel=True)
        
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
            compact_raw_fold(DATASET_DIR, fold, normalize_audio=False, fold_percentage=FOLD_PERCENTAGE, resample_to=22050)
            REGENERATE_FOLD[fold-1] = False
        generate_compacted_fold_spectrograms(DATASET_DIR, fold, 
                                                sample_rate=22050, spectrogram_bands=128, 
                                                hop_length=512, log_mel=True)
        
        with code_timer("load_compacted_dataset", debug=True):
            audio_meta, audio_raw, audio_spectrograms_raw = load_raw_compacted_dataset(DATASET_DIR, folds = fold_list)


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