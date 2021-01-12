import torch
import pandas as pd
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
        pass

    #lista data, ogni elemento della lista è
    #un dizionario con campi : filepath,classeId,className,
    #                           metadata= dizionario con altri dati
   
    def load_dataset(self,sample):
        csvData = pd.read_csv('UrbanSound8K/metadata/UrbanSound8K.csv')
        sounds = list()
        Dict dizionario = dict()
        for i in range(10):
          print(csvData.iloc[0, :])
          dizionario.add(csvData[i])
            
      

         
        
#TODO EMANUELE
        prep_sample = sample
        return prep_sample
    
#TODO MICHELE
        pass

if __name__ == "__main__":
    DATASET_DIR = "UrbanSound8K"#data
    DATASET_NAME = "UrbanSound8K"
    dataset = SoundDataset(DATASET_DIR, DATASET_NAME)

    
   