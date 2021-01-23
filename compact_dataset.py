
import os
from utils.timing import function_timer, code_timer
from utils.dataset_utils import compact_urbansound_dataset, load_compacted_dataset, generate_compacted_fold_spectrograms
from utils.spectrogram_utils import display_heatmap, generate_mel_spectrogram_librosa, compute_spectrogram_frames

if __name__=="__main__":

    base_dir = os.path.dirname(os.path.realpath(__file__))
    DATASET_DIR = os.path.join(base_dir,"data")
    print(DATASET_DIR)

    #If folds have already been generated, there is no need to do it again

    #audio_meta, audio_raw_mm, spectro_mm = compact_urbansound_dataset(DATASET_DIR,folds = [1,2])
    #audio_meta, audio_raw_mm, spectro_mm = load_compacted_dataset(DATASET_DIR,folds = [1,2])

    #for i, audio in enumerate(audio_raw):
    #   print("{}: {}".format(i,audio))
    #audio_meta, audio_raw_mm, spectro_mm = compact_urbansound_dataset(DATASET_DIR, folds = [1])
    #audio_meta, audio_raw_mm, spectro_mm = compact_urbansound_dataset(DATASET_DIR, folds = [2])
    #audio_meta, audio_raw_mm, spectro_mm = compact_urbansound_dataset(DATASET_DIR, folds = [3])
    #audio_meta, audio_raw_mm, spectro_mm = compact_urbansound_dataset(DATASET_DIR, folds = [4])
    #audio_meta, audio_raw_mm, spectro_mm = compact_urbansound_dataset(DATASET_DIR, folds = [5])
    #audio_meta, audio_raw_mm, spectro_mm = compact_urbansound_dataset(DATASET_DIR, folds = [6])
    #audio_meta, audio_raw_mm, spectro_mm = compact_urbansound_dataset(DATASET_DIR, folds = [7])
    #audio_meta, audio_raw_mm, spectro_mm = compact_urbansound_dataset(DATASET_DIR, folds = [8])
    #audio_meta, audio_raw_mm, spectro_mm = compact_urbansound_dataset(DATASET_DIR, folds = [9])
    #audio_meta, audio_raw_mm, spectro_mm = compact_urbansound_dataset(DATASET_DIR, folds = [10], spectrogram_bands=128)

    #audio_meta, audio_raw_mm, spectro_mm = generate_compacted_fold_spectrograms(DATASET_DIR, folds = [1], spectrogram_bands=128)
    #audio_meta, audio_raw_mm, spectro_mm = generate_compacted_fold_spectrograms(DATASET_DIR, folds = [2], spectrogram_bands=128)
    #audio_meta, audio_raw_mm, spectro_mm = generate_compacted_fold_spectrograms(DATASET_DIR, folds = [3], spectrogram_bands=128)
    #audio_meta, audio_raw_mm, spectro_mm = generate_compacted_fold_spectrograms(DATASET_DIR, folds = [4], spectrogram_bands=128)
    #audio_meta, audio_raw_mm, spectro_mm = generate_compacted_fold_spectrograms(DATASET_DIR, folds = [5], spectrogram_bands=128)
    #audio_meta, audio_raw_mm, spectro_mm = generate_compacted_fold_spectrograms(DATASET_DIR, folds = [6], spectrogram_bands=128)
    #audio_meta, audio_raw_mm, spectro_mm = generate_compacted_fold_spectrograms(DATASET_DIR, folds = [7], spectrogram_bands=128)
    #audio_meta, audio_raw_mm, spectro_mm = generate_compacted_fold_spectrograms(DATASET_DIR, folds = [8], spectrogram_bands=128)
    #audio_meta, audio_raw_mm, spectro_mm = generate_compacted_fold_spectrograms(DATASET_DIR, folds = [9], spectrogram_bands=128)
    #audio_meta, audio_raw_mm, spectro_mm = generate_compacted_fold_spectrograms(DATASET_DIR, folds = [10], spectrogram_bands=128)
    #with code_timer("load_compacted_dataset", debug=True):
    #    audio_meta, audio_raw, audio_spectrograms = load_compacted_dataset(DATASET_DIR,folds = [10], spectrogram_bands=128)

    #for i, audio in enumerate(audio_raw):
        #print("{}: {}".format(i,audio))
        #if audio_spectrograms is not None:
        #   display_heatmap(audio_spectrograms[i])