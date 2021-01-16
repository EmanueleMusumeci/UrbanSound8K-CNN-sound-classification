import torch
from torch import nn
from torchsummary import summary
"""
#Implementazione Originale
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=24, kernel_size=5, padding=0)
        self.conv2 = nn.Conv2d(in_channels=24, out_channels=36, kernel_size=4, padding=0)
        self.conv3 = nn.Conv2d(in_channels=36, out_channels=48, kernel_size=3, padding=0)

        self.fc1 = nn.Linear(in_features=48, out_features=60)
        self.fc2 = nn.Linear(in_features=60, out_features=10)

        #self.criterion = nn.CrossEntropyLoss()
        #self.optimizer = optim.Adam(self.parameters(), lr=0.001, eps=1e-07, weight_decay=1e-3)

        #self.device = device

    def forward(self, x):
        # cnn layer-1
        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=(3,3), stride=3)
        x = F.relu(x)

        # cnn layer-2
        x = self.conv2(x)
        x = F.max_pool2d(x, kernel_size=(2,2), stride=2)
        x = F.relu(x)

        # cnn layer-3
        x = self.conv3(x)
        x = F.relu(x)

        # global average pooling 2D
        x = F.avg_pool2d(x, kernel_size=x.size()[2:])
        x = x.view(-1, 48)

        # dense layer-1
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5)

        # dense output layer
        x = self.fc2(x)

        return x


"""

import torch
from torch import nn
from torchsummary import summary
#From Emanuele Vision and Perception:

class ConvolutionalLayer(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size=(1,1), kernel_stride = (1,1), ReLU = None, use_batch_normalization = True, padding=0):
    super(ConvolutionalLayer, self).__init__()

    assert isinstance(kernel_size, tuple) and len(kernel_size) == 2, "Wrong kernel size: "+str(kernel_size)
    assert isinstance(kernel_stride, tuple) and len(kernel_stride) == 2, "Wrong kernel stride: "+str(kernel_stride)

    self.relu = ReLU
    self.use_batch_normalization = use_batch_normalization

    self.kernel_size = kernel_size
    self.kernel_stride = kernel_stride

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size, stride=kernel_stride, padding=padding) #we don't need bias as data is normalized
    if self.use_batch_normalization: self.bn = nn.BatchNorm2d(self.out_channels)

  def forward(self, x):
    x = self.conv(x)
    if self.use_batch_normalization: 
      x = self.bn(x)
    if self.relu is not None:
      x = self.relu(x)
    return x

class DenseLayer(nn.Module):
  def __init__(self, in_dim, out_dim, ReLU = None):
    super(DenseLayer, self).__init__()
    self.relu = ReLU

    self.in_dim = in_dim
    self.out_dim = out_dim

    self.dense = nn.Linear(in_dim, out_dim)

  def forward(self, x):
    x = self.dense(x)
    if self.relu is not None: x = self.relu(x)

    return x

##Modello nuovo usando classe ConvolutionalLayer:
#per figura dell' architettura vedere https://github.com/mariostrbac/environmental-sound-classification sezione CNN Model
class ConvolutionalNetwork(nn.Module):
    def __init__(self, input_size, use_ReLU=True, use_batch_normalization=True, dropout_p=0.5):
        super(ConvolutionalNetwork, self).__init__()
        
        self.in_channels = input_size[-1]

        self.use_ReLU = use_ReLU
        self.use_batch_normalization = use_batch_normalization

        self.dropout_p = dropout_p
        
        #Conv 1:
        #in_channels =1
        self.conv_layer_1 = ConvolutionalLayer(self.in_channels, 24, kernel_size=(5,5), kernel_stride=(1,1), ReLU = None, use_batch_normalization = False)
        #Conv 2:
        self.conv_layer_2 = ConvolutionalLayer(24, 36, kernel_size=(5,5), kernel_stride=(1,1), ReLU = None, use_batch_normalization = False)
        #Conv 3:
        self.conv_layer_3 = ConvolutionalLayer(36, 48, kernel_size=(5,5), kernel_stride=(1,1), ReLU = None, use_batch_normalization = False)

        #Linear 1:

        self.dense_1 = DenseLayer(48,60)

        #Linear 2:

        self.dense_2 = DenseLayer(60,10)

    def forward(self, x):
        # cnn layer-1
        x = self.conv_layer_1(x)
        x = F.max_pool2d(x, kernel_size=(3,3), stride=3)
        x = F.relu(x)

        # cnn layer-2
        x = self.conv_layer_2(x)
        x = F.max_pool2d(x, kernel_size=(2,2), stride=2)
        x = F.relu(x)

        # cnn layer-3
        x = self.conv_layer_3(x)
        x = F.relu(x)

        # global average pooling 2D
        x = F.avg_pool2d(x, kernel_size=x.size()[2:])
        x = x.view(-1, 48)

        # dense layer-1
        x = self.dense_1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5)

        # dense output layer
        x = self.dense_2(x)

        return x

if __name__ == "__main__":
    
    net = ConvolutionalNetwork(1)
    print(net)