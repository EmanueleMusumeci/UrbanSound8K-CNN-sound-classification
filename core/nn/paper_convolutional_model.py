import torch
from torch import nn
from torchsummary import summary

'''
CNN model used in the paper https://arxiv.org/pdf/1608.04363v2.pdf
'''
class PaperConvolutionalNetwork(nn.Module):
    def __init__(self, input_size, dropout_p = 0.5):
        super(PaperConvolutionalNetwork, self).__init__()
        
        self.in_channels = input_size[-1]

        self.dropout_p = dropout_p

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
        
            nn.Dropout(p = self.dropout_p),
            nn.Linear(2400, 64),
            nn.ReLU(),

            nn.Dropout(p = self.dropout_p),
            nn.Linear(64, 10)
        )


    def forward(self, x):
        
        x = x.permute(0, 3, 1, 2)
        
        x = self.convolutional_layers(x)
        
        x = self.flatten(x)
        
        x = self.dense_layers(x)

        return x


