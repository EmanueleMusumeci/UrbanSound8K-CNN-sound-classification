import os

import numpy as np

try:
    from utils.timing import function_timer, code_timer
    from utils.model_utils import unpickle_data
except:
    pass 

#from https://github.com/karolpiczak/paper-2015-esc-convnet/blob/master/Code/_Datasets/Setup.ipynb
@function_timer
def load_compacted_dataset(dataset_dir, folds = []):
    """Load raw audio and metadata content from the UrbanSound8K dataset."""
    
    audio_meta = []
    audio_raw = []
    for fold in folds:
        if os.path.isfile(os.path.join(dataset_dir,'urban_meta_fold_{}.pkl'.format(fold))) and os.path.isfile(os.path.join(dataset_dir,'urban_audio_fold_{}.dat'.format(fold))):
            fold_meta = unpickle_data(os.path.join(dataset_dir,'urban_meta_fold_{}.pkl'.format(fold)))
            fold_raw = np.memmap(os.path.join(dataset_dir,'urban_audio_fold_{}.dat'.format(fold)), dtype='float32', mode='r', shape=(len(fold_meta), 88200))
            audio_meta = np.concatenate((audio_meta,fold_meta))
            audio_raw.append(fold_raw)
        else:
            raise FileNotFoundError

    audio_raw = np.vstack(audio_raw)
    return audio_meta, audio_raw


#from https://github.com/karolpiczak/paper-2015-esc-convnet/blob/master/Code/_Datasets/Setup.ipynb
def compact_urbansound_dataset(dataset_dir, folds = [1,2,3,4,5,6,7,8,9,10], skip_first_line = True):
    """Load raw audio and metadata content from the UrbanSound8K dataset and generate a .dat file"""
    total_files = 0
    for fold in folds:
        total_files += len([name for name in os.listdir(os.path.join(dataset_dir, "UrbanSound8K", "audio", "fold"+str(fold))) if os.path.isfile(os.path.join(dataset_dir, "UrbanSound8K", "audio", "fold"+str(fold),name))])
    
    with open(os.path.join(dataset_dir,'UrbanSound8K','metadata','UrbanSound8K.csv'), 'r') as read_obj:
        csv_reader = reader(read_obj)
        
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

                audio_file, sr = load_audio_file(audiodict["file_path"], 4000)

                audiodict["sampling_rate"] = sr
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
        mm = np.memmap(os.path.join(dataset_dir,'urban_audio_fold_{}.dat'.format(fold)), dtype='float32', mode='w+', shape=(len(audio_raw_stacked), 88200))
        mm[:] = audio_raw_stacked[:]
    mm.flush()
    del audio_raw
    return audio_meta, mm

if __name__=="__main__":

    base_dir = os.path.dirname(os.path.realpath(__file__))
    DATASET_DIR = os.path.join(base_dir,"data")
    print(DATASET_DIR)
    #audio_meta, mm = compact_urbansound_dataset(DATASET_DIR,folds = [1,2])
    #audio_meta, audio_raw = load_compacted_dataset(DATASET_DIR,folds = [1,2])

    #for i, audio in enumerate(audio_raw):
    #   print("{}: {}".format(i,audio))
    #audio_meta, mm = compact_urbansound_dataset(DATASET_DIR,folds = [1])
    #audio_meta, mm = compact_urbansound_dataset(DATASET_DIR,folds = [2])
    #audio_meta, mm = compact_urbansound_dataset(DATASET_DIR,folds = [3])
    #audio_meta, mm = compact_urbansound_dataset(DATASET_DIR,folds = [4])
    #audio_meta, mm = compact_urbansound_dataset(DATASET_DIR,folds = [5])
    #audio_meta, mm = compact_urbansound_dataset(DATASET_DIR,folds = [6])
    #audio_meta, mm = compact_urbansound_dataset(DATASET_DIR,folds = [7])
    #audio_meta, mm = compact_urbansound_dataset(DATASET_DIR,folds = [8])
    #audio_meta, mm = compact_urbansound_dataset(DATASET_DIR,folds = [9])
    #audio_meta, mm = compact_urbansound_dataset(DATASET_DIR,folds = [10])
    #audio_meta, audio_raw = load_compacted_dataset(DATASET_DIR,folds = [2,3,4,5,6,7,8,9,10])
    #with code_timer("load_compacted_dataset", debug=self.debug_preprocessing):
    #    audio_meta, audio_raw = load_compacted_dataset(DATASET_DIR,folds = [2])

    #for i, audio in enumerate(audio_raw):
    #    print("{}: {}".format(i,audio))

