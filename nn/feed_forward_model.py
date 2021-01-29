import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import os
import librosa

try:
    from Dataset import SoundDatasetFold
    from DataLoader import DataLoader
    from data_augmentation.image_transformations import *
except:
    pass

#https://github.com/jaron/deep-listening/blob/master/1-us8k-ffn-extract-explore.ipynb
def load_sound_files(parent_dir, file_paths):
    raw_sounds = []
    for fp in file_paths:
        X,sr = librosa.load(parent_dir + fp)
        raw_sounds.append(X)
    return raw_sounds
#traduzione in PyTorch da tensorflow del tutorial:
#   https://github.com/jaron/deep-listening/blob/master/2-us8k-ffn-train-predict.ipynb

class FeedForwardNetwork(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size):
        super(FeedForwardNetwork, self).__init__()                    
        
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.relu2 = nn.ReLU()
        #1d perchè lavoro con un array
        self.drop1 = nn.Dropout(p=0.5)#aggiungi p a init variables?
        self.fc3 = nn.Linear(self.hidden_size, output_size)
        self.relu3 = nn.ReLU()
        self.drop2 = nn.Dropout(p=0.5)#aggiungi p a init variables?


    def forward(self, mfccs, chroma, mel ,contrast, tonnetz):
        x = torch.cat([mfccs, chroma, mel, contrast, tonnetz],1)
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.drop1(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.drop2(out)

        return out


if __name__ == "__main__":
 
    base_dir = os.path.dirname(os.path.realpath(__file__))
    DATASET_DIR = os.path.join(base_dir,"data")
    DATASET_NAME = "UrbanSound8K"

    selected_classes = [0,1,2,3,4,5,6,7,8,9]

    spectrogram_frames_per_segment = 41
    spectrogram_bands = 60

    CNN_INPUT_SIZE = (spectrogram_bands, spectrogram_frames_per_segment)

    right_shift_transformation = SpectrogramShift(input_size=CNN_INPUT_SIZE,width_shift_range=4,shift_prob=0.9)
    left_shift_transformation = SpectrogramShift(input_size=CNN_INPUT_SIZE,width_shift_range=4,shift_prob=0.9, left=True)
    random_side_shift_transformation = SpectrogramShift(input_size=CNN_INPUT_SIZE,width_shift_range=4,shift_prob=0.9, random_side=True)

    background_noise_transformation = SpectrogramAddGaussNoise(input_size=CNN_INPUT_SIZE,prob_to_have_noise=0.55)

    dataset = SoundDatasetFold(DATASET_DIR, DATASET_NAME, 
                                folds = [1], 
                                shuffle = True, 
                                generate_spectrograms = False, 
                                shift_transformation = right_shift_transformation, 
                                background_noise_transformation = background_noise_transformation, 
                                audio_augmentation_pipeline = [], 
                                spectrogram_frames_per_segment = spectrogram_frames_per_segment, 
                                spectrogram_bands = spectrogram_bands, 
                                compute_deltas=True, 
                                compute_delta_deltas=True, 
                                test = False, 
                                progress_bar = True,
                                selected_classes=selected_classes
                                )

    batch_size = 128

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    batch = next(iter(dataloader))
   
    nn = FeedForwardNetwork(154, 256, dataset.get_num_classes())

    mfccs = batch["mfccs"]
    chroma = batch["chroma"]
    mel = batch["mel"]
    contrast = batch["contrast"]
    tonnetz = batch["tonnetz"]

    output = nn(mfccs, chroma, mel, contrast, tonnetz)
    print("output_net: ",output)
    print(output.shape)
    












































#tutorial FFN: 
#https://medium.com/biaslyai/pytorch-introduction-to-neural-network-feedforward-neural-network-model-e7231cff47cb

#ESEMPIO FUNZIONANTE E PIU' PROFONDO, MIGLIORA (DIMINUISCE) TEST LOSS
"""
class FeedForwardNetwork(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(FeedForwardNetwork, self).__init__()                      # Inherited from the parent class nn.Module
        self.input_size = input_size
        self.hidden_size  = hidden_size

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.relu3 = nn.ReLU()
        self.fc4 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.relu4 = nn.ReLU()
        self.fc5 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.relu5 = nn.ReLU()
        self.fc6 = torch.nn.Linear(self.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        out = self.relu4(out)
        out = self.fc5(out)
        out = self.relu5(out)
        out = self.fc6(out)
        out = self.sigmoid(out)
        return out
"""
"""
############################# to test ##################################
# Per dataset qua sotto uso Sigmoid perchè classificazione binaria
# Per nostro problema invece Softmax

def blob_label(y, label, loc): # assign labels
    target = np.copy(y)
    for l in loc:
        target[y == l] = label
    return target

x_train, y_train = make_blobs(n_samples=40, n_features=2, cluster_std=1.5, shuffle=True)
x_train = torch.FloatTensor(x_train)
y_train = torch.FloatTensor(blob_label(y_train, 0, [0]))
y_train = torch.FloatTensor(blob_label(y_train, 1, [1,2,3]))
x_test, y_test = make_blobs(n_samples=10, n_features=2, cluster_std=1.5, shuffle=True)
x_test = torch.FloatTensor(x_test)
y_test = torch.FloatTensor(blob_label(y_test, 0, [0]))
y_test = torch.FloatTensor(blob_label(y_test, 1, [1,2,3]))

#print("-----")
#print(x_train)
#print("-----")
#print(y_train)
#print("-----")
#print(x_test)
#print("-----")
#print(y_test)

net = FeedForwardNetwork(2, 10)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(net.parameters(), lr = 0.01)
print(net)
#print(optimizer)

net.eval()

y_pred = net(x_test)
#print(y_pred)
before_train = criterion(y_pred.squeeze(), y_test)
print('Test loss before training' , before_train.item())

#new weights can be learned after every epoch
net.train()
epoch = 20
for epoch in range(epoch):
    #sets the gradients to zero before we start backpropagation. 
    # PyTorch accumulates the gradients from the backward passes from the previous epochs
    optimizer.zero_grad()
    # Forward pass
    y_pred = net(x_train)
    # Compute Loss
    loss = criterion(y_pred.squeeze(), y_train)
   
    #print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
    # Backward pass ,  computes the gradients
    loss.backward()
    #update the weights
    optimizer.step()

net.eval()
y_pred = net(x_test)
after_train = criterion(y_pred.squeeze(), y_test) 
print('Test loss after Training' , after_train.item())
"""