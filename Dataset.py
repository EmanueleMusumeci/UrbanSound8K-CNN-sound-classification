import torch

class SoundDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, dataset_name):
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name

#TODO MICHELE
        self.data = self.load_dataset(dataset_dir)

#TODO MICHELE
    def __getitem__(self, index):
        #Decidere come salvare dati -> estrarre sample
        #sample = <qualcosa>
        
        preprocessed_sample = self.preprocess(sample)

        #Considerare quali labels usare
        return preprocessed_sample, labels 

    def preprocess(self, sample):
#TODO EMANUELE
        prep_sample = sample
        return prep_sample
    
    def __len__(self):
#TODO MICHELE
        pass

if __name__ == "__main__":
    DATASET_DIR = "data"
    DATASET_NAME = "UrbanSound8K"
    dataset = UrbanSoundDataset(DATASET_DIR, DATASET_NAME)

