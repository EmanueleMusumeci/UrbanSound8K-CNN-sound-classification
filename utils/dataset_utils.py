import os
import math

import numpy as np

import csv 
import json

from tqdm import tqdm
try:
    from utils.timing import function_timer, code_timer
    from utils.model_utils import unpickle_data, pickle_data
    from utils.audio_utils import load_audio_file
    from utils.spectrogram_utils import generate_mel_spectrogram_librosa, display_heatmap, compute_spectrogram_frames
except Exception as e:
    raise e
    pass 

def save_compacted_data(data, file_name, shape):
    mm = np.memmap(file_name, dtype='float32', mode='w+', shape=shape)
    mm[:] = data[:]
    mm.flush()

    return mm

def load_compacted_data(file_name, shape):
    mm = np.memmap(file_name, dtype="float32", mode="r", shape=shape)
    return mm


#from https://github.com/karolpiczak/paper-2015-esc-convnet/blob/master/Code/_Datasets/Setup.ipynb
@function_timer
def load_raw_compacted_dataset(dataset_dir, folds = [], spectrogram_bands = 128, log_mel=True, sample_rate = 22050, hop_length = 512, seconds = 4):
    """Load raw audio and metadata content from the UrbanSound8K dataset."""
    
    spectro_frames = compute_spectrogram_frames(seconds, sample_rate, hop_length)

    audio_meta = []
    audio_raw = []
    audio_spectrograms_raw = []
    
    #Look for a index.json file, that should have already been generated upon compacting the dataset
    assert os.path.exists(os.path.join(dataset_dir,"index.json")), "Please compact dataset first to generate the index.json file"
    
    #This dictionary will be used to index the preprocessed files
    with open(os.path.join(dataset_dir,"index.json"), mode="r") as f:
        json_index = json.load(f)
        
    for fold_number in folds:
        #1) Check that fold has already been compacted
        assert "fold_"+str(fold_number) in json_index.keys(), "Fold {} not yet compacted".format(fold_number)
        assert "raw" in json_index["fold_"+str(fold_number)].keys(), "Fold {} not yet compacted".format(fold_number)

        #2) Load fold meta-data
        assert "meta_data_file_name" in json_index["fold_"+str(fold_number)].keys(), "Meta-data not found for fold {}".format(fold_number)
        print(json_index["fold_"+str(fold_number)]["meta_data_file_name"])
        fold_meta = unpickle_data(json_index["fold_"+str(fold_number)]["meta_data_file_name"])
        audio_meta = np.concatenate((audio_meta,fold_meta))

        #3) Load fold audio data
        #Load audio data
        audio_length = json_index["fold_"+str(fold_number)]["raw"]["audio_length"]
        fold_raw = np.memmap(json_index["fold_"+str(fold_number)]["raw"]["audio_file_name"], dtype='float32', mode='r', shape=(len(fold_meta), audio_length))
        audio_raw.append(fold_raw)

        #Load spectrograms (if they exist)
        if "spectrograms" in json_index["fold_"+str(fold_number)]["raw"].keys():
            audio_spec_file_name = json_index["fold_"+str(fold_number)]["raw"]["spectrograms"]["file_name"]
            audio_spec_bands = json_index["fold_"+str(fold_number)]["raw"]["spectrograms"]["bands"]
            audio_spec_frames = json_index["fold_"+str(fold_number)]["raw"]["spectrograms"]["frames"]

            audio_spec = np.memmap(audio_spec_file_name, dtype="float32", mode = "r", shape=(len(fold_meta), audio_spec_bands, audio_spec_frames))
            audio_spectrograms_raw.append(audio_spec)

    audio_raw = np.vstack(audio_raw)

    if len(audio_spectrograms_raw)>0:
        audio_spectrograms_raw = np.vstack(audio_spectrograms_raw)

    return audio_meta, audio_raw, audio_spectrograms_raw

#from https://github.com/karolpiczak/paper-2015-esc-convnet/blob/master/Code/_Datasets/Setup.ipynb
@function_timer
def load_preprocessed_compacted_dataset(dataset_dir, preprocessing_name, folds = [], spectrogram_bands = 128, log_mel=True, sample_rate = 22050, hop_length = 512, seconds = 4):
    """Load raw audio and metadata content from the UrbanSound8K dataset."""
    
    spectro_frames = compute_spectrogram_frames(seconds, sample_rate, hop_length)

    audio_meta = []
    audio_preprocessed = {}
    audio_spectrograms_preprocessed = {}
    
    assert isinstance(preprocessing_name, str), "Preprocessing name should be a string"

    #Look for a index.json file, that should have already been generated upon compacting the dataset
    assert os.path.exists(os.path.join(dataset_dir,"index.json")), "Please compact dataset first to generate the index.json file"
    
    #This dictionary will be used to index the preprocessed files
    with open(os.path.join(dataset_dir,"index.json"), mode="r") as f:
        json_index = json.load(f)

    for fold_number in folds:
        #1) Check that fold has already been compacted
        assert "fold_"+str(fold_number) in json_index.keys(), "Fold {} not yet compacted".format(fold_number)
        assert "preprocessed" in json_index["fold_"+str(fold_number)].keys(), "Fold {} not yet preprocessed".format(fold_number)
        assert preprocessing_name in json_index["fold_"+str(fold_number)]["preprocessed"].keys(), "Fold {} not yet preprocessed with preprocessing '{}'".format(fold_number, preprocessing_name)

        #2) Load fold meta-data
        assert "meta_data_file_name" in json_index["fold_"+str(fold_number)].keys(), "Meta-data not found for fold {}".format(fold_number)
        fold_meta = unpickle_data(json_index["fold_"+str(fold_number)]["meta_data_file_name"])
        audio_meta = np.concatenate((audio_meta,fold_meta))

        #3) Load fold audio data
        #Load preprocessed audio data and spectrograms (if they exist)
        for entry in json_index["fold_"+str(fold_number)]["preprocessed"][preprocessing_name]:
            preprocessing_value = entry["value"]
            file_name = entry["audio_file_name"]
            audio_length = entry["audio_length"]
            
            assert os.path.exists(file_name), "Could not find preprocessed fold {}".format(file_name)

            fold_preprocessed = np.memmap(file_name, dtype='float32', mode='r', shape=(len(fold_meta), audio_length))
            
            if preprocessing_value not in audio_preprocessed.keys():
                audio_preprocessed[preprocessing_value] = []

            audio_preprocessed[preprocessing_value].append(fold_preprocessed)

            if "spectrograms" in entry.keys():
                audio_spec_preprocessed_file_name = entry["spectrograms"]["file_name"]
                audio_spec_preprocessed_bands = entry["spectrograms"]["bands"]
                audio_spec_preprocessed_frames = entry["spectrograms"]["frames"]

                audio_spec_preprocessed = np.memmap(audio_spec_preprocessed_file_name, dtype="float32", mode = "r", shape=(len(fold_meta), audio_spec_bands, audio_spec_preprocessed_frames))
                
                if preprocessing_value not in audio_preprocessed.keys():
                    audio_spectrograms_preprocessed[preprocessing_value] = []
                
                audio_spectrograms_preprocessed[preprocessing_value].append(audio_spec_preprocessed)

    for preprocessing_value, preprocessed_clips in audio_preprocessed.items():
        preprocessed_clips = np.vstack(preprocessed_clips)

    for preprocessing_value, preprocessed_spectrograms in audio_spectrograms_preprocessed.items():
        preprocessed_spectrograms = np.vstack(preprocessed_spectrograms)
    
    return audio_meta, audio_preprocessed, audio_spectrograms_preprocessed

#from https://github.com/karolpiczak/paper-2015-esc-convnet/blob/master/Code/_Datasets/Setup.ipynb
def compact_raw_fold(dataset_dir, fold_number, resample_to = 22050, skip_first_line = True, duration_seconds = 4):
    """Load raw audio and metadata content from the UrbanSound8K dataset and generate a .dat file"""
    DURATION = duration_seconds * 1000

    #Count files in fold
    total_files = len([name for name in os.listdir(os.path.join(dataset_dir, "UrbanSound8K", "audio", "fold"+str(fold_number))) if os.path.isfile(os.path.join(dataset_dir, "UrbanSound8K", "audio", "fold"+str(fold_number),name))])
    
    #Parse the csv dataset index
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
        audio_meta = []
        #Dictionary (int->list) fold -> raw audio data list
        audio_raw = []

        print("Compacting fold {} ({} files)...".format(fold_number, total_files))
        progress_bar = tqdm(total=total_files, desc="Sample", position=0)
        for i, audio in enumerate(audios_data_from_csv):
            if i>100: break
            if int(audio[5]) == fold_number:
                
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
                
                audio_file, sr = load_audio_file(audiodict["file_path"], duration = DURATION)
                
                if sr!=resample_to:
                    print("File {} (sampling rate: {}) resampled to {}".format(audiodict["file_path"],sr,resample_to))
                    audio_file = librosa.resample(audio_file, orig_sr=sr, target_sr=resample_to)

                audiodict["sampling_rate"] = sr
                
                audio_meta.append(audiodict)
                audio_raw.append(audio_file)
                progress_bar.update(1)

    progress_bar.close()
    
    if not os.path.exists(os.path.join(dataset_dir, "raw")):
        os.makedirs(os.path.join(dataset_dir, "raw"))

    #Pickle meta-data
    print("Saving meta-data...")
    meta_data_file_name = os.path.join(dataset_dir,"raw","urban_meta_fold_{}.pkl".format(fold_number))
    pickle_data(audio_meta, meta_data_file_name)

    #Save compacted audio
    print("Compacting raw audio...")
    audio_raw_stacked = np.vstack(audio_raw)

    audio_file_name = os.path.join(dataset_dir,"raw",'urban_audio_fold_{}.dat'.format(fold_number))
    audio_mm = save_compacted_data(audio_raw_stacked, audio_file_name, shape=(len(audio_raw_stacked), 88200))

    #Free up unused memory
    del audio_raw

    #Load index.json or create it if it does not exist
    if os.path.exists(os.path.join(dataset_dir, "index.json")):
        try:
            with open(os.path.join(dataset_dir, "index.json"), "r") as f:
                json_index = json.load(f)
            print("Updating index.json ...")
        except:
            json_index = {}
            print("Creating index.json ...")
    else:
        json_index = {}
    
    #Update index.json
    json_index["fold_"+str(fold_number)] = {"meta_data_file_name": meta_data_file_name}
    json_index["fold_"+str(fold_number)]["raw"] = {"audio_file_name" : audio_file_name, "audio_length" : 88200}
        
    #Save the updated index.json
    with open(os.path.join(dataset_dir,"index.json"), mode="w") as f:
        json.dump(json_index, f, indent=4)

    print("Finished compacting fold {}!".format(fold_number))
    return audio_meta, audio_mm

def generate_fold_spectrograms(fold_meta, fold_audio, spectrogram_bands = 128, log_mel=True, 
                                sample_rate=22050, hop_length=512, seconds = 4):
    
    spectrograms = []
    progress_bar = tqdm(total=len(fold_meta), desc="Sample", position=0)
    for i, audio_raw in enumerate(fold_audio):
        spectro = np.array(generate_mel_spectrogram_librosa(audio_raw, spectrogram_bands, log_mel = log_mel, debug_time_label=""))
        spectrograms.append(spectro)
        progress_bar.update(1)
    progress_bar.close()

    #Ensures that the big bulk of data is removed from memory
    del audio_raw

    return spectrograms

def generate_compacted_fold_spectrograms(dataset_dir, fold_number, spectrogram_bands = 128, log_mel=True, 
                                         sample_rate=22050, hop_length=512, seconds = 4, preprocessing_name=None):

    spectro_frames = compute_spectrogram_frames(seconds, sample_rate, hop_length)

    if preprocessing_name is not None:
        assert isinstance(preprocessing_name, str), "Preprocessing name should be a string"

    #Look for a index.json file, that should have already been generated upon compacting the dataset
    assert os.path.exists(os.path.join(dataset_dir,"index.json")), "Please compact dataset first to generate the index.json file"
    
    #This dictionary will be used to index the preprocessed files
    with open(os.path.join(dataset_dir,"index.json"), mode="r") as f:
        json_index = json.load(f)

    #1) Check that fold has already been compacted
    assert "fold_"+str(fold_number) in json_index.keys(), "Compact fold {} before generating spectrograms".format(fold_number)

    #2) Load fold meta-data
    assert "meta_data_file_name" in json_index["fold_"+str(fold_number)].keys(), "Meta-data not found for fold {}".format(fold_number)
    fold_meta = unpickle_data(json_index["fold_"+str(fold_number)]["meta_data_file_name"])
    
    if preprocessing_name is not None:
        #3) Check that fold has already been preprocessed with this preprocessing before 
        assert "preprocessed" in json_index["fold_"+str(fold_number)].keys() \
            and preprocessing_name in json_index["fold_"+str(fold_number)]["preprocessed"].keys(),\
                 "Preprocess compacted fold {} with preprocessing '{}' before generating spectrograms".format(fold_number, preprocessing_name) 
            
        print("Generating spectrograms for fold {} ({} files) preprocessed with {}...".format(fold_number, len(fold_meta), preprocessing_name))
        
        #4) Extract all preprocessing values applied for this preprocessing
        preprocessing_values = []
        preprocessed_file_names = []
        for entry in json_index["fold_"+str(fold_number)]["preprocessed"][preprocessing_name]:
            preprocessing_value = entry["value"]
            preprocessed_file_names = entry["audio_file_name"]

            print("\tPreprocessing value: {}".format(preprocessing_value))
            #5) For each preprocessing value load the preprocessed fold, generate spectrograms from it and then save them
            
            #Load fold
            fold_audio = load_compacted_data(file_name, shape=(len(fold_meta), spectrogram_bands, spectro_frames))

            #Generate spectrograms from this fold
            fold_spectrograms = generate_fold_spectrograms(fold_meta, fold_audio)

            #3) TODO Compact and save generated spectrograms
            print("\t\tCompacting...")
            fold_spectrograms = np.array(fold_spectrograms)
            print("\t\tSpectrograms shape: {}".format(fold_spectrograms.shape))
            
            file_name = os.path.join(dataset_dir,"preprocessed",preprocessing_name,"urban_spectrograms_fold_{}_seconds_{}_bands_{}_sr_{}_hop_{}{}_preprocessed_{}_{}.dat".format(fold, seconds, spectrogram_bands, sample_rate, hop_length, ("_log_mel" if log_mel else ""), preprocessing_name, str(preprocessing_value).replace(".","_")))    
            entry["spectrograms"] = {"file_name": file_name, "bands" : spectrogram_bands, "seconds": seconds, "frames" : spectro_frames, "hop_length" : hop_length, "sample_rate" : sample_rate}
            
            fold_spectrograms = save_compacted_data(fold_spectrograms, file_name, shape=(fold_spectrograms.shape[0],fold_spectrograms.shape[1],fold_spectrograms.shape[2]))   

            #write out the new entry to preprocessing_index.json
            with open(os.path.join(dataset_dir,"index.json"), mode="w") as f:
                json.dump(json_index, f, indent=4)

    else:
        #3) Check that fold has already been compacted before 
        assert "raw" in json_index["fold_"+str(fold_number)].keys(), \
                "Compact fold {} before generating spectrograms".format(fold_number) 
            
        print("Generating spectrograms for fold {} ({} files)...".format(fold_number, len(fold_meta)))
        
        #4) Retrieve fold file name
        fold_file_name = json_index["fold_"+str(fold_number)]["raw"]["audio_file_name"]
        fold_audio_length = json_index["fold_"+str(fold_number)]["raw"]["audio_length"]

        #5) Load compacted fold and generate spectrograms from it
        
        #Load fold
        fold_audio = load_compacted_data(fold_file_name, shape=(len(fold_meta), fold_audio_length))

        #Generate spectrograms from this fold
        print(fold_meta)
        print(fold_audio.shape)
        fold_spectrograms = generate_fold_spectrograms(fold_meta, fold_audio)

        #6) Compact and save generated spectrograms
        print("\t\tCompacting...")
        fold_spectrograms = np.array(fold_spectrograms)
        print("\t\tSpectrograms shape: {}".format(fold_spectrograms.shape))
        
        #Compacted spectrograms file name
        file_name = os.path.join(dataset_dir,"raw","urban_spectrograms_fold_{}_seconds_{}_bands_{}_sr_{}_hop_{}{}.dat".format(fold_number, seconds, spectrogram_bands, sample_rate, hop_length, ("_log_mel" if log_mel else "")))
        json_index["fold_"+str(fold_number)]["raw"]["spectrograms"] = {"file_name": file_name, "bands" : spectrogram_bands, "seconds": seconds, "frames" : spectro_frames, "hop_length" : hop_length, "sample_rate" : sample_rate}
        
        fold_spectrograms = save_compacted_data(fold_spectrograms, file_name, shape=(fold_spectrograms.shape[0],fold_spectrograms.shape[1],fold_spectrograms.shape[2]))   

        #write out the new entry to preprocessing_index.json
        with open(os.path.join(dataset_dir,"index.json"), mode="w") as f:
            json.dump(json_index, f, indent = 4)
    
    return fold_spectrograms

'''
Given metadata and raw audio for a fold, preprocesses it, generating a compacted preprocessed audio fold for each preprocessing factor
used in the preprocessor 
(Example: Performing PitchShift using values [-2, -1, 1], then three preprocessed folds will be generated, one for
each one of these values)
'''
def generate_compacted_preprocessed_fold(dataset_dir, fold_number, preprocessing_name, preprocessor):

    spectro_frames = compute_spectrogram_frames(seconds, sample_rate, hop_length)
    
    #Extract the used values from the preprocessor
    preprocessing_values = preprocessor.values

    #0) Preliminary checks
    assert isinstance(preprocessing_name, str), "Preprocessing name should be a string"

    #Look for a index.json file, that should have already been generated upon compacting the dataset
    assert os.path.exists(os.path.join(dataset_dir,"index.json")), "Please compact dataset first to generate the index.json file"
    
    #1) Load index.json and perform some more checks
    #This dictionary will be used to index the preprocessed files
    with open(os.path.join(dataset_dir,"index.json"), mode="r") as f:
        json_index = json.load(f)

    #Check that fold has already been compacted
    assert "fold_"+str(fold_number) in json_index.keys(), "Compact fold {} before generating spectrograms".format(fold_number)
    assert "raw" in json_index["fold_"+str(fold_number)].keys(), \
            "Compact fold {} before generating spectrograms".format(fold_number) 

    #2) Load fold meta-data
    assert "meta_data_file_name" in json_index["fold_"+str(fold_number)].keys(), "Meta-data not found for fold {}".format(fold_number)
    fold_meta = unpickle_data(json_index["fold_"+str(fold_number)]["meta_data_file_name"])
    
    #Create or overwrite index entry for this preprocessing pipeline
    json_index["fold_"+str(fold_number)]["preprocessed"][preprocessing_name] = [] 

    print("Preprocessing fold: {}".format(fold_number))
    
    #3) Retrieve fold file name
    fold_file_name = json_index["fold_"+str(fold_number)]["raw"]["audio_file_name"]
    fold_audio_length = json_index["fold_"+str(fold_number)]["raw"]["audio_length"]

    #4) Load compacted fold and generate spectrograms from it
    #Load fold
    fold_audio = load_compacted_data(fold_file_name, shape=(len(fold_meta), fold_audio_length))

    preprocessed_clips = []
    #Preprocess all raw audio clips in current fold
    for i, value in enumerate(preprocessing_values):
        progress_bar = tqdm(total=len(fold_meta)*len(preprocessing_values), desc="Preprocessing samples (value: {})".format(value), position=0)
        print("Generating preprocessed clips for fold ({} files) with preprocessing '{}' (value: {})".format(len(fold_meta), preprocessing_name, value))
        for i, (audio_meta, audio_raw) in enumerate(zip(fold_meta,fold_raw)):

            prep_clip = preprocessor(audio_raw, value = value)
            preprocessed_clips.append(prep_clip)

            progress_bar.update(1)
        progress_bar.close()

        print("Compacting...")

        preprocessed_clips = np.array(preprocessed_clips)
        print("Stacked shape: {}".format(preprocessed_clips.shape))
        
        file_name = os.path.join(dataset_dir, "preprocessed", preprocessing_name, "urban_audio_preprocessed_fold_{}_{}_{}.dat".format(fold, preprocessing_name, str(i)))
        
        preprocessed_clips = save_compacted_data(preprocessed_clips, file_name, shape=(len(preprocessed_clips),preprocessed_clips.shape[1]))

        #Generate an entry for the JSON index file
        json_index["fold_"+str(fold_number)]["preprocessed"][preprocessing_name].append({"value": preprocessing_value, "audio_file_name": file_name, "audio_length": preprocessed_clips.shape[1]})
    
        #write out the new entry to preprocessing_index.json
        with open(os.path.join(dataset_dir,"index.json"), mode="w") as f:
            json.dump(json_index, f, indent = 4)
        
        #Ensures that the big bulk of data is removed from memory
        del audio_meta
        del audio_raw

    return preprocessed_clips
        
#TODO nel dataset: scelta casuale dell'augmentation 

#TODO metodo che, per ogni fold: 
# 1) genera i file del fold e gli spettrogrammi (non preprocessati) 
# 2) Genera i file audio preprocessati 
# 3) Genera da essi gli spettrogrammi
#NB USARE LE FUNZIONI GIA CREATE