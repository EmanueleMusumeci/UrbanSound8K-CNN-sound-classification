import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from sklearn.datasets import make_blobs
#tutorial: 
#https://medium.com/biaslyai/pytorch-introduction-to-neural-network-feedforward-neural-network-model-e7231cff47cb
class FeedForwardNetwork(nn.Module):

    def __init__(self,input_size, hidden_size):
        super(FeedForwardNetwork, self).__init__()
        self.input_size = input_size
        #layer 1
        self.hidden_size1 = hidden_size
        self.fc1 = nn.Linear(self.input_size,self.hidden_size1)
        self.relu1 = nn.ReLU()
        #layer 2
        self.hidden_size2 = hidden_size
        self.fc2 = nn.Linear(self.hidden_size1,self.hidden_size2)
        self.relu2 = nn.ReLU()
        #layer 3
        self.fc3 = nn.Linear(self.hidden_size2, 1)
        #self.softmax = nn.Softmax(dim=1)
        self.sigmoid = torch.nn.Sigmoid()
        
    
    def forward(self, x):
        hidden1 = self.fc1(x)
        relu1 = self.relu(hidden1)
        hidden2 = self.fc2(relu1)
        relu2 = self.relu(hidden2)
        hidden3 = self.fc3(relu2)
        #output = self.softmax(hidden3)
        output = self.sigmod(output)
        return output
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

print("-----")
print(x_train)
print("-----")
print(y_train)
print("-----")
print(x_test)
print("-----")
print(y_test)

net = FeedForwardNetwork(2, 10)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(),lr=0.001)
print(net)
print(optimizer)

net.eval()

y_pred = net(x_test)
print(y_pred)
before_train = criterion(y_pred.squeeze(), y_test)
print('Test loss before training' , before_train.item())
