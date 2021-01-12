import torch
import librosa
from matplotlib import pyplot as plt

def load_sound_file(file_path, sr=22050, mono=True):
    #Loads the raw sound time series and returns also the sampling rate
    raw_sound, sr = librosa.load(file_path)
    return raw_sound

'''
Displays a wave plot for the input raw sound (using the Librosa library)
'''
def plot_sound_waves(sound, sound_file_name = None, show=False, sound_class=None, sr=22050):
    plot_title = "Wave plot"
    
    if sound_file_nameis not None:
        plot_title += "File: "+sound_file_name
    
    if sound_class is not None:
        plot_title+=" (Class: "+sound_class+")"
    
    plot = plt.figure(plot_title)
    librosa.display.waveplot(np.array(sound),sr=sr)
    plt.title(plot_title)
    
    if show:
        plt.show()

def plot_sound_spectrogram(sound, sound_file_name = None, log_scale = False, show = False, sr = 22050, use_matplotlib_render=False):
    plot_title = "Spectrogram"
    
    if sound_file_nameis not None:
        plot_title += "File: "+sound_file_name
    
    if sound_class is not None:
        plot_title+=" (Class: "+sound_class+")"
    
    if use_matplotlib_render:
        plt.specgram(np.array(sound), Fs = sr)
    else:
        libros.display.specshow(sound, x_axis="Time (sec)", y_axis="Log frequency (kHz)")

    plt.title(plot_title)

    if show:
        plt.show()

class SoundDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, dataset_name):
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name

#TODO MICHELE
        self.data = self.load_dataset(dataset_dir)

#TODO EMANUELE
    def __getitem__(self, index):
        #Decidere come salvare dati -> estrarre sample
        sample = self.data[index]
        class_id = sample["class_id"]
        class_name = sample["class_name"]
        meta_data = sample["meta_data"]
        
        preprocessed_sample = self.preprocess(sample)

        #Considerare quali labels usare
        return preprocessed_sample, class_id, class_name, meta_data 

    def preprocess(self, sample):
#TODO EMANUELE
        prep_sample = sample
        return prep_sample
    
    def __len__(self):
        return len(self.data)

if __name__ == "__main__":
    DATASET_DIR = "data"
    DATASET_NAME = "UrbanSound8K"
    dataset = UrbanSoundDataset(DATASET_DIR, DATASET_NAME)
    print(dataset[0])


