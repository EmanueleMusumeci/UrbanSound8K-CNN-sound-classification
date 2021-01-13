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

        self.fields = [
                       "image",                             #[batch_size, CNN_input_width, CNN_input_height]
                       "region_proposals_bounding_boxes"    #List([N_bounding_box_of_image, 4]) of size batch_size
                      ]

        self.gold_fields = [
                            "gold_class_names",
                            "region_proposals_class_names",
                            "gold_bounding_boxes"
                           ]

        self.unpreprocessed_fields = [
                                      "image_id",
                                      "image_name",
                                      "original_image"
                                     ]

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

        bounding_boxes = []
        for field, value in current_batch.items():
          if field in self.unpreprocessed_fields:
            continue
          # We don't encode the sentences as we need them raw
          # (the model will receive only the embeddings)
          if value is None: continue

          array = []

          #Convert to numpy arrays 
          try:
            if field=="gold_class_names" or field=="region_proposals_class_names":
              if self.dataset.encode_labels:
                #array = np.asarray(value, dtype=np.int64)
              #else:
                continue
            else:
              array = np.asarray(value, dtype=np.int64)
          except Exception as e:
              print("Exception during conversion to numpy arrays, raised by field ",field," with value ",value)
              for i,el in enumerate(value):
                print(current_batch["image_name"][i])
                print(el.shape)
              raise e

          #Convert to tensors
          try:
            #If we use Bert embeddings use a float tensor            
            if field=="region_proposals_bounding_boxes" or field=="gold_bounding_boxes":
              for i,element in enumerate(array):
                current_batch[field][i] = torch.autograd.Variable(
                      torch.from_numpy(element)).type(torch.LongTensor).cpu()
            #elif field=="gold_bounding_boxes" or field=="gold_class_names" or field=="region_proposals_class_names":
            #  continue
            elif field=="image": 
              current_batch[field] = torch.autograd.Variable(
                  torch.from_numpy(array)).type(torch.FloatTensor).cpu()
            elif field=="region_proposal_class_names" or field=="gold_class_names":
              current_batch[field] = torch.autograd.Variable(
                  torch.from_numpy(array)).type(torch.LongTensor).cpu()

          except Exception as e:
              print("Exception during conversion to tensors, raised by field ",field," with value ",value)
              raise e

        if self.shuffle:
          array = self.shuffle_batch(array, batch_size, random_indices=permutation_mask)

        if not self.batch_first:
            current_batch[field] = torch.transpose(current_batch[field], 0, 1)

        return current_batch

#TODO RAW (UNANNOTATED) IMAGE PREDICTION PIPELINE
    '''
    def __call__(self, input_image):
      images, = self.dataset(input_image)
      batches = []
      for sentence in sentences:
        batches.append(self.collate_sample([sentence]))
      return batches
    

    def collate_sample(self, samples):
      #This function is called only when in prediction mode (in the docker framework)

      #If in prediction mode and the predicate_id is -1, this sentence has no
      #predicate so we return a batch of empty lists (it will not be predicted anyway)
      
      if self.prediction_mode and samples[0]["predicate_id"]==-1:
        sample_batch = self.collate_fn(samples)
        return sample_batch
      else:
        sample_batch = self.collate_fn(samples)
        sample_batch = self.preprocess_batch(sample_batch)
        return sample_batch
    '''

    def get_random_image(self, only_multi_object=False, only_single_object=False):
      sample = [self.dataset.get_random_image(only_multi_object, only_single_object)]
      
      return self.preprocess_batch(self.collate_fn(sample))

    def preprocess_image_from_dataset_id(self, id):
      return self.preprocess_batch(self.collate_fn(self.dataset[id]))

    def pad_sequences(self, batch, max_sequence_length, padding_symbol_index=None, padding_dimension=1, lengths_vector=None, nested = False, pad_left=False, use_token_index_as_padding_symbol=False, debug=False):
        """
        Adds padding to all sequences of a batch that are shorter than the longest sequence in that batch
        Args:
          - batch: list of sequences to pad
          - max_sequence_length: length of the longest sentence in the batch, if a sentence
            is shorter than this length it is padded in order for it to reach this length
          - padding_symbol_index: symbol used as padding
          OPTIONAL:
          - padding_dimension: if 1 only a single padding_symbol is added, if >1 instead
                      of a padding symbol, a whole vector of padding symbols of length padding_dimension is added
          - lengths_vector: vector of lengths of the batch vector, that will be used
            to pack the sequences in the neural network. For every padding added a 0
            will be added to this vector
          - nested: determines if we want to pad a list of lists or a list
          - pad_left: determines wether we want to pad to the left or to the right
          - use_token_index_as_padding_symbol
        """
        assert padding_symbol_index is not None or use_token_index_as_padding_symbol, "Specify a padding symbol!"
        if nested:
            for line in batch:
                for i, sequence in enumerate(line):
                    if len(sequence) < max_sequence_length:
                        for j in range(max_sequence_length - len(sequence)):
                            if use_token_index_as_padding_symbol:
                              padding_symbol_index = len(sequence)+j
                            if padding_dimension > 1:
                              if pad_left:
                                sequence.insert(0,[padding_symbol_index]
                                                * padding_dimension)
                              else:
                                sequence.append([padding_symbol_index]
                                                * padding_dimension)
                            else:
                              if pad_left:
                                sequence.insert(0,padding_symbol_index)
                              else:
                                sequence.append(padding_symbol_index)
                            if lengths_vector is not None:
                                lengths_vector[i].append(0)
        else:
            for i, sequence in enumerate(batch):
                if debug:
                    print(len(sequence))
                    print(max_sequence_length)
                if len(sequence) < max_sequence_length:
                    for j in range(max_sequence_length - len(sequence)):
                        if use_token_index_as_padding_symbol:
                          padding_symbol_index = len(sequence)+j
                        if padding_dimension > 1:
                          if pad_left:
                            sequence.insert(0,[padding_symbol_index]
                                            * padding_dimension)
                          else:
                            sequence.append([padding_symbol_index]
                                            * padding_dimension)
                        else:
                          if pad_left:
                            sequence.insert(0,padding_symbol_index)
                          else:
                            sequence.append(padding_symbol_index)
                        if lengths_vector is not None:
                            lengths_vector[i].append(0)
                

    def get_padding_masks(self, sentence_lengths, max_sentence_length):
      '''Returns a binary padding mask (has 0 for padding tokens, 1 for the others)
         for the sentences in the batch
      '''
      padding_mask = []
      for length in sentence_lengths:
          line = [1]*max_sentence_length
          line[:length] = [0]*length
          padding_mask.append(line)
      return padding_mask

    def shuffle_batch(self, batch, batch_dim, random_indices=None):
      if random_indices is None:
        random_indices = np.random.permutation(batch_dim)
      batch = batch[random_indices]

      return batch