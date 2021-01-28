
import os
from utils.timing import function_timer, code_timer
from utils.dataset_utils import *
from utils.spectrogram_utils import display_heatmap, generate_mel_spectrogram_librosa, compute_spectrogram_frames
from utils.audio_utils import play_sound
from data_augmentation.audio_transformations import *


if __name__=="__main__":

#TODO Test with multiple folds
    base_dir = os.path.dirname(os.path.realpath(__file__))
    DATASET_DIR = os.path.join(base_dir,"data")
    print(DATASET_DIR)

    #delete_compacted_dataset(DATASET_DIR)

    #fold_list = [1]
    fold_list = [1,2,3,4,5,6,7,8,9,10]

    REGENERATE_RAW_FOLDS = False
    
    audio_augmentations = [
                           #PitchShift(values = [-2, -1, 1, 2]), \
                           PitchShift(values = [-3.5, -2.5, 2.5, 3.5]),
                           MUDADynamicRangeCompression(),
                           BackgroundNoise({
                               "street_scene_1" : "150993__saphe__street-scene-1.wav",
                               "street_scene_3" : "173955__saphe__street-scene-3.wav",
                               "street_valencia" : "207208__jormarp__high-street-of-gandia-valencia-spain.wav",
                               "city_park_tel_aviv" : "268903__yonts__city-park-tel-aviv-israel.wav",
                           }, files_dir = os.path.join(DATASET_DIR, "UrbanSound8K-JAMS", "background_noise")),
                           TimeStretch(values = [0.81, 0.93, 1.07, 1.23])
                           
                           ]
    
    for audio_augmentation in audio_augmentations:
        for fold in fold_list:
            #compact_raw_fold(DATASET_DIR, fold, only_first_n_samples=4)
            if not os.path.exists(os.path.join(DATASET_DIR, "raw", "urban_audio_fold_"+str(fold)+".dat")) or REGENERATE_RAW_FOLDS:
                compact_raw_fold(DATASET_DIR, fold)
            generate_compacted_fold_spectrograms(DATASET_DIR, fold)

            generate_compacted_preprocessed_fold(DATASET_DIR, fold, audio_augmentation.name, audio_augmentation)
            generate_compacted_fold_spectrograms(DATASET_DIR, fold, preprocessing_name=audio_augmentation.name)
        
        with code_timer("load_compacted_dataset", debug=True):
            audio_meta, audio_raw, audio_spectrograms_raw = load_raw_compacted_dataset(DATASET_DIR, folds = fold_list)

        with code_timer("load_compacted_preprocessed_dataset", debug=True):
            audio_meta, audio_preprocessed, audio_spectrograms_preprocessed = load_preprocessed_compacted_dataset(DATASET_DIR, preprocessing_name=audio_augmentation.name, folds = fold_list)

    
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