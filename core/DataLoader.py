import os

import math
import numpy as np
import torch

from core.Dataset import SoundDatasetFold
from core.data_augmentation.image_transformations import SpectrogramAddGaussNoise, SpectrogramReshape, SpectrogramShift


'''
Implements a custom DataLoader to do the following (see the __iter__ method for
a detailed descriptions of all the operations involved in data preparation)
Args:
  - dataset: iterable dataset that provides the samples
  OPTIONAL:
  - batch_size: (default: 1)
  - batch_first: determines if the batch is batch-major or time-major (default: True)
  - device
  - tensorize_gold_data: allows preventing gold data from being turned into tensor form
  - padding_symbol_index

'''
class DataLoader():
    def __init__(self,
                 dataset,
                 batch_size=1,
                 batch_first=True,
                 shuffle=False,
                 device="cuda",
                 tensorize_gold_data = False,
                 verbose = False,
                 ):

        self.dataset = dataset
        self.batch_size = batch_size      
        self.batch_first = batch_first
        self.shuffle = shuffle
        self.device = device
        self.tensorize_gold_data = tensorize_gold_data

        self.fields = self.dataset.get_preprocessed_fields()

        self.gold_fields = self.dataset.get_gold_fields()

        self.unpreprocessed_fields = self.dataset.get_unpreprocessed_fields()

    #NOTICE: len was not implemented in this instance because the dataset length is not always predictable
    # (depending on the window sampling methods chosen) 
    '''
    def __len__(self):
      return math.ceil(len(self.dataset)/self.batch_size)

    def __getitem__(self, index):
      batch_n = math.ceil(len(self.dataset)/self.batch_size)
      if(index>=batch_n): raise StopIteration
      samples = []
      for i in range(self.batch_size*index,self.batch_size*(index+1)):
        try:
            samples.append(self.dataset[i])
        except StopIteration:
            break
      batch = self.collate_fn(samples)
      batch = self.preprocess_batch(batch)
      return batch
    '''
    
    def __call__(self, sample_meta, preprocessing_value = None):
      samples = self.dataset(sample_meta, preprocessing_value = preprocessing_value)
      batch = self.collate_fn(samples)
      batch = self.preprocess_batch(batch)
      return batch

    '''
    __iter__ method for this custom DataLoader class, prepares data by performing
    the following operations:
      Collation function:
      1) Collects a batch of self.batch_size samples from the Dataset instance
      2) Collates and preprocesses this batch (turns fields into tensors)
    '''
    def __iter__(self):
      dataset_iterator = iter(self.dataset)
      j = 0
      samples = []
      while True:
        if j==self.batch_size:
          batch = self.collate_fn(samples)
          batch = self.preprocess_batch(batch)
          
          yield batch
          samples = []
          j=0
        try:
            samples.append(next(dataset_iterator))
            j+=1

        except StopIteration:
          if len(samples)>0:
            batch = self.collate_fn(samples)
            batch = self.preprocess_batch(batch)
            
            yield batch
          else:
            yield None

          break
        

    def collate_fn(self, samples):
        current_batch = {}
        #Populate batch with empty fields
        for field in self.fields:
          current_batch[field] = []

        for gold_field in self.gold_fields:
          current_batch[gold_field] = []

        for unpreprocessed_field in self.unpreprocessed_fields:
          current_batch[unpreprocessed_field] = []

        # Populate the batch with samples
        for sample in samples:
          for field in self.fields:
            current_batch[field].append(sample[field])

          for unpreprocessed_field in self.unpreprocessed_fields:
            current_batch[unpreprocessed_field].append(sample[unpreprocessed_field])

          # This distinction is necessary to use a different batch structure when
          # gold data is not provided (like when doing a batched prediction)
          for gold_field in self.gold_fields:
            current_batch[gold_field].append(sample[gold_field])

        return current_batch

    def preprocess_batch(self, current_batch):
        # Find max sentence length for sequence padding
        #max_bounding_boxes_per_image = max([len(bounding_boxes) for bounding_boxes in current_batch["bounding_boxes"]])

        # Apply padding to sentences and for every padding appended, append a 0 in the length vector
        # (that will be used in the neural network to correctly pack the sequence)
        
        if self.shuffle:
          # Use the same permutation mask for all batches
          permutation_mask = np.random.permutation(batch_size)

        for field, value in current_batch.items():
          if field in self.unpreprocessed_fields:
            continue

          array = []

          #Convert to numpy arrays 
          try:
            if field=="class_id":
                array = np.asarray(value, dtype=np.int32)
            elif field=="original_spectrogram" or field=="preprocessed_spectrogram":
                array = np.asarray(value, dtype=np.float32)
            else:
                array = np.asarray(value)
          except Exception as e:
              print("Exception during conversion to numpy arrays, raised by field ",field," with value ",value)
              for i,el in enumerate(value):
                print(el.shape)
              raise e

          #Convert to tensors
          try:        
            if field=="class_id": 
              current_batch[field] = torch.autograd.Variable(
                  torch.from_numpy(array)).type(torch.LongTensor).cpu()
            else:
              current_batch[field] = torch.autograd.Variable(
                  torch.from_numpy(array)).type(torch.FloatTensor).cpu()

          except Exception as e:
              print("Exception during conversion to tensors, raised by field ",field," with value ",value)
              raise e

        if self.shuffle:
          array = self.shuffle_batch(array, batch_size, random_indices=permutation_mask)

        if not self.batch_first:
            current_batch[field] = torch.transpose(current_batch[field], 0, 1)

        return current_batch

    def shuffle_batch(self, batch, batch_dim, random_indices=None):
      if random_indices is None:
        random_indices = np.random.permutation(batch_dim)
      batch = batch[random_indices]

      return batch

    '''
    NOT USED - NOT TESTED
    def parse_batch(batch,batch_size):
        for i in range(batch_size):
            mfccs = batch["mfccs"][i]
            chroma = batch["chroma"][i]
            mel = batch["mel"][i]
            contrast = batch["contrast"][i]
            tonnetz = batch["tonnetz"][i]
            print(mfccs,chroma,mel,contrast,tonnetz)
            ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
            print(ext_features)
            features = np.vstack([features,ext_features])
            print(features)
            labels = np.append(labels, fn.split('fold')[1].split('-')[1])
            print(labels)

    def parse_audio_files():
        features, labels = np.empty((0,193)), np.empty(0)
        #for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
        for i in range(8):
            try:
                mfccs = batch["mfccs"][i]
                chroma = batch["chroma"][i]
                mel = batch["mel"][i]
                contrast = batch["contrast"][i]
                tonnetz = batch["tonnetz"][i]
                ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
                features = np.vstack([features,ext_features])
                labels = np.append(labels, fn.split('fold')[1].split('-')[1])
            except:
                print("Error processing " + fn + " - skipping")
        return np.array(features), np.array(labels, dtype = np.int)   
    '''