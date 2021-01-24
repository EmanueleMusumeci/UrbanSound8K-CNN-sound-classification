
import os
from utils.timing import function_timer, code_timer
from utils.dataset_utils import *
from utils.spectrogram_utils import display_heatmap, generate_mel_spectrogram_librosa, compute_spectrogram_frames

if __name__=="__main__":

    base_dir = os.path.dirname(os.path.realpath(__file__))
    DATASET_DIR = os.path.join(base_dir,"data")
    print(DATASET_DIR)

    fold_list = [1]
    #fold_list = [1,2,3,4,5,6,7,8,9,10]

    for fold in fold_list:
        compact_raw_fold(DATASET_DIR, fold)
        generate_compacted_fold_spectrograms(DATASET_DIR, fold)
        generate_compacted_preprocessed_fold(DATASET_DIR, fold, "PS1", )

    with code_timer("load_compacted_dataset", debug=True):
        audio_meta, audio_raw, audio_spectrograms_raw = load_raw_compacted_dataset(DATASET_DIR, folds = fold_list, spectrogram_bands=128)

    for i, audio in enumerate(audio_raw):
        print("{}: {} ({})".format(i,audio, len(audio)))
    #    assert len(audio) == 88200
        if audio_spectrograms_raw is not None:
            print(audio_spectrograms_raw.shape)
            display_heatmap(audio_spectrograms_raw[i])