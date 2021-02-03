import torch
from torch import nn
from torchsummary import summary

##Modello nuovo usando classe ConvolutionalLayer:
#per figura dell' architettura vedere https://github.com/mariostrbac/environmental-sound-classification sezione CNN Model
class PaperConvolutionalNetwork(nn.Module):
    def __init__(self, input_size):
        super(PaperConvolutionalNetwork, self).__init__()
        
        self.in_channels = input_size[-1]

        self.convolutional_layers = nn.Sequential(
            #Conv 1:
            nn.Conv2d(self.in_channels, 24, kernel_size=(5,5), stride=(1,1)),
            nn.MaxPool2d(kernel_size = (4,2), stride=(4,2)),
            nn.ReLU(),
            
            #Conv 2:
            nn.Conv2d(24, 48, kernel_size=(5,5), stride=(1,1)),
            nn.MaxPool2d(kernel_size = (4,2), stride=(4,2)),
            nn.ReLU(),
            
            #Conv 3:
            nn.Conv2d(48, 48, kernel_size=(5,5), stride=(1,1)),
            nn.ReLU()
        )

        self.flatten = nn.Flatten()

        self.dense_layers = nn.Sequential(
        
            nn.Dropout(0.5),
            nn.Linear(2400, 64),
            nn.ReLU(),

            nn.Dropout(0.5),
            nn.Linear(64, 10)
        )


    def forward(self, x):
        
        x = x.permute(0, 3, 1, 2)
        
        x = self.convolutional_layers(x)
        
        x = self.flatten(x)
        
        x = self.dense_layers(x)

        return x


