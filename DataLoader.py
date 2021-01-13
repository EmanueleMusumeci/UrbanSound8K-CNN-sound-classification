import math
import numpy as np
import torch

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

        self.dataset = dataset
        self.batch_size = batch_size      
        self.batch_first = batch_first
        self.shuffle = shuffle
        self.sequence_padding = sequence_padding
        self.padding_vector = padding_vector
        self.padding_symbol_index = padding_symbol_index
        self.device = device
        self.tensorize_gold_data = tensorize_gold_data

        self.fields = self.dataset.get_preprocessed_fields()

        self.gold_fields = self.dataset.get_gold_fields()

        self.unpreprocessed_fields = self.dataset.get_unpreprocessed_fields()

        self.prediction_mode = prediction_mode

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

    def __iter__(self):
      '''
      __iter__ method for this custom DataLoader class, prepares data by performing
      the following operations:
        Collation function:
        1) Collects a batch of self.batch_size samples from the Dataset instance
        Preprocessing function:
        1) Detect the maximum sentence length in the batch
        2) Pad all other sentences up to the same length
        3) Optionally, shuffle samples in the batch
        4) Initialize tensors from sample components on the correct device
        5) Optionally, if time-major format is requested, transpose the first two dimensions
          of the tensors in the batch
      '''
      num_batches = math.ceil(len(self.dataset) / self.batch_size)

      dataset_iterator = iter(self.dataset)
      for i in range(num_batches):
        #print("batch: "+str(i))
        samples = []
        for j in range(self.batch_size):
          try:
              samples.append(next(dataset_iterator))
          except StopIteration:
              break
        
        batch = self.collate_fn(samples)
        batch = self.preprocess_batch(batch)
        
        yield batch

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
            if field=="original_spectrogram" or field=="preprocessed_spectrogram":
                array = np.asarray(value, dtype=np.int64)
            else:
                array = np.asarray(value)
          except Exception as e:
              print("Exception during conversion to numpy arrays, raised by field ",field," with value ",value)
              for i,el in enumerate(value):
                print(current_batch["image_name"][i])
                print(el.shape)
              raise e

          #Convert to tensors
          try:        
            if field=="original_spectrogram" or field=="preprocessed_spectrogram": 
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