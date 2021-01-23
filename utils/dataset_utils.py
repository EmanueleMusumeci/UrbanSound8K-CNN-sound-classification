import os
import math

import numpy as np

import csv 

from tqdm import tqdm
try:
    from utils.timing import function_timer, code_timer
    from utils.model_utils import unpickle_data, pickle_data
    from utils.audio_utils import load_audio_file
    from utils.spectrogram_utils import generate_mel_spectrogram_librosa, display_heatmap, compute_spectrogram_frames
except Exception as e:
    raise e
    pass 

#from https://github.com/karolpiczak/paper-2015-esc-convnet/blob/master/Code/_Datasets/Setup.ipynb
@function_timer
def load_compacted_dataset(dataset_dir, folds = [], spectrogram_bands = 128, log_mel=True, sample_rate = 22050, hop_length = 512, seconds = 4):
    """Load raw audio and metadata content from the UrbanSound8K dataset."""
    
    audio_meta = []
    audio_raw = []
    audio_spectrograms = []
    spectro_frames = compute_spectrogram_frames(seconds, sample_rate, hop_length)
    for fold in folds:
        if os.path.isfile(os.path.join(dataset_dir,'urban_meta_fold_{}.pkl'.format(fold))) and os.path.isfile(os.path.join(dataset_dir,'urban_audio_fold_{}.dat'.format(fold))):
            fold_meta = unpickle_data(os.path.join(dataset_dir,'urban_meta_fold_{}.pkl'.format(fold)))
            fold_raw = np.memmap(os.path.join(dataset_dir,'urban_audio_fold_{}.dat'.format(fold)), dtype='float32', mode='r', shape=(len(fold_meta), 88200))
            audio_meta = np.concatenate((audio_meta,fold_meta))
            audio_raw.append(fold_raw)
            spectro_file_name = "urban_meta_fold_{}_spectrograms_{}_seconds_{}_bands_{}_sr_{}_hop{}.dat".format(fold, seconds, spectrogram_bands, sample_rate, hop_length, ("_log_mel" if log_mel else ""))
            if spectrogram_bands is not None and os.path.isfile(os.path.join(dataset_dir,spectro_file_name)):
                audio_spec = np.memmap(os.path.join(dataset_dir,spectro_file_name), dtype="float32", mode = "r", shape=(len(fold_meta), spectrogram_bands, spectro_frames))
                audio_spectrograms.append(audio_spec)
        else:
            raise FileNotFoundError

    audio_raw = np.vstack(audio_raw)
    if spectrogram_bands is not None:
        audio_spectrograms = np.vstack(audio_spectrograms)
    return audio_meta, audio_raw, audio_spectrograms


#from https://github.com/karolpiczak/paper-2015-esc-convnet/blob/master/Code/_Datasets/Setup.ipynb
def compact_urbansound_dataset(dataset_dir, folds = [1,2,3,4,5,6,7,8,9,10], resample_to = 22050, skip_first_line = True, 
                                spectrogram_bands = None, log_mel=True, sample_rate=22050, hop_length=512, seconds = 4):
    """Load raw audio and metadata content from the UrbanSound8K dataset and generate a .dat file"""
    DURATION = seconds * 1000

    total_files = 0
    for fold in folds:
        total_files += len([name for name in os.listdir(os.path.join(dataset_dir, "UrbanSound8K", "audio", "fold"+str(fold))) if os.path.isfile(os.path.join(dataset_dir, "UrbanSound8K", "audio", "fold"+str(fold),name))])
    
    with open(os.path.join(dataset_dir,'UrbanSound8K','metadata','UrbanSound8K.csv'), 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        
        #next skips the first line that contains the header info
        if skip_first_line: 
            next(csv_reader)

        audios_data_from_csv = []
        for row in csv_reader:
            audios_data_from_csv.append(row)

        audio_ids = []
        #Dictionary (int->list) fold -> audio file info list
        audio_meta = {}
        #Dictionary (int->list) fold -> raw audio data list
        audio_raw = {}
        audio_spectro = {}

        print("Compacting {} folds ({} files)...".format(len(folds), total_files))
        progress_bar = tqdm(total=total_files, desc="Sample", position=0)
        for i, audio in enumerate(audios_data_from_csv):
            fold_number = int(audio[5])
            if fold_number in folds:
                
                metadata = {
                    "fsID":audio[1],
                    "start":audio[2],
                    "end":audio[3],
                    "salience":audio[4]
                }   
                audiodict = {
                    "file_path": os.path.join(dataset_dir, "UrbanSound8K", "audio", "fold"+str(fold_number), audio[0]),
                    "class_id":int(audio[6]),
                    "class_name":audio[7],
                    "meta_data": metadata,
                }
                
                if fold_number not in audio_meta.keys():
                    audio_meta[fold_number] = []    
                    audio_raw[fold_number] = []    
                    audio_spectro[fold_number] = []

                audio_file, sr = load_audio_file(audiodict["file_path"], duration = DURATION)
                
                if sr!=resample_to:
                    print("File {} (sampling rate: {}) resampled to {}".format(audiodict["file_path"],sr,resample_to))
                    audio_file = librosa.resample(audio_file, orig_sr=sr, target_sr=resample_to)

                audiodict["sampling_rate"] = sr

                if spectrogram_bands is not None:
                    audio_spectrogram = np.array(generate_mel_spectrogram_librosa(audio_file, spectrogram_bands, log_mel = log_mel, debug_time_label=""))
                    #display_heatmap(audio_spectrogram)
                    audio_spectro[fold_number].append(audio_spectrogram)
                
                audio_meta[fold_number].append(audiodict)
                audio_raw[fold_number].append(audio_file)
                progress_bar.update(1)

    progress_bar.close()

    print("Saving meta-data...")
    for fold, meta_data in audio_meta.items():
        pickle_data(meta_data, os.path.join(dataset_dir,"urban_meta_fold_{}.pkl".format(fold)))

    print("Compacting raw audio...")
    for fold, audio_clips in audio_raw.items():
        audio_raw_stacked = np.vstack(audio_clips)
        spectrograms_mm = np.memmap(os.path.join(dataset_dir,'urban_audio_fold_{}.dat'.format(fold)), dtype='float32', mode='w+', shape=(len(audio_raw_stacked), 88200))
        spectrograms_mm[:] = audio_raw_stacked[:]
    spectrograms_mm.flush()
    del audio_raw

    if spectrogram_bands is not None:
        print("Compacting spectrograms...")
        for fold, audio_spectrograms in audio_spectro.items():
            audio_spectro_stacked = np.array(audio_spectrograms)

            spectrogram_name = "urban_meta_fold_{}_spectrograms_{}_seconds_{}_bands_{}_sr_{}_hop{}.dat".format(fold, seconds, spectrogram_bands, sample_rate, hop_length, ("_log_mel" if log_mel else ""))

            audio_mm = np.memmap(os.path.join(dataset_dir,spectrogram_name), dtype='float32', mode='w+', shape=(len(audio_spectro_stacked),audio_spectro_stacked.shape[1],audio_spectro_stacked.shape[2]))
            audio_mm[:] = audio_spectro_stacked[:]
        
        audio_mm.flush()
        del audio_spectrograms
    return audio_meta, audio_mm, spectrograms_mm

def generate_compacted_fold_spectrograms(dataset_dir, folds = [1,2,3,4,5,6,7,8,9,10], spectrogram_bands = None, log_mel=True, 
                                         sample_rate=22050, hop_length=512, seconds = 4):
    """Load raw audio and metadata content from the UrbanSound8K dataset and generate a .dat file"""
    DURATION = seconds * 1000

    audio_spectro = {}

    spectro_frames = compute_spectrogram_frames(seconds, sample_rate, hop_length)
    for fold in folds:

        audio_spectro[fold] = []

        print("Loading fold {}...".format(fold))
        if os.path.isfile(os.path.join(dataset_dir,'urban_meta_fold_{}.pkl'.format(fold))) and os.path.isfile(os.path.join(dataset_dir,'urban_audio_fold_{}.dat'.format(fold))):
            fold_meta = unpickle_data(os.path.join(dataset_dir,'urban_meta_fold_{}.pkl'.format(fold)))
            fold_raw = np.memmap(os.path.join(dataset_dir,'urban_audio_fold_{}.dat'.format(fold)), dtype='float32', mode='r', shape=(len(fold_meta), 88200))
            
            print("Generating spectrograms for fold ({} files)...".format(len(folds), len(fold_meta)))
            progress_bar = tqdm(total=len(fold_meta), desc="Sample", position=0)
            for i, (audio_meta, audio_raw) in enumerate(zip(fold_meta,fold_raw)):
                audio_spectrogram = np.array(generate_mel_spectrogram_librosa(audio_raw, spectrogram_bands, log_mel = log_mel, debug_time_label=""))
                audio_spectro[fold].append(audio_spectrogram)
                progress_bar.update(1)
            progress_bar.close()
        else:
            raise FileNotFoundError


        print("Compacting spectrograms...")
        for fold, audio_spectrograms in audio_spectro.items():
            print("Compacting spectrograms for fold {}...".format(fold))
            audio_spectro_stacked = np.array(audio_spectrograms)        
            spectrogram_name = "urban_meta_fold_{}_spectrograms_{}_seconds_{}_bands_{}_sr_{}_hop{}.dat".format(fold, seconds, spectrogram_bands, sample_rate, hop_length, ("_log_mel" if log_mel else ""))
            
            print(audio_spectro_stacked.shape)
            
            spectrograms_mm = np.memmap(os.path.join(dataset_dir,spectrogram_name), dtype='float32', mode='w+', shape=(len(audio_spectro_stacked),audio_spectro_stacked.shape[1],audio_spectro_stacked.shape[2]))
            spectrograms_mm[:] = audio_spectro_stacked[:]
        spectrograms_mm.flush()
        del audio_spectrograms
    return audio_meta, audio_raw, spectrograms_mm 
