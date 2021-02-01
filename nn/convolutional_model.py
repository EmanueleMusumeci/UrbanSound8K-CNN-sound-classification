import torch
from torch import nn
from torchsummary import summary


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
class CustomConvolutionalNetwork(nn.Module):
    def __init__(self, input_size, use_ReLU=True, use_batch_normalization=True, dropout_p=0.5):
        super(CustomConvolutionalNetwork, self).__init__()
        
        self.in_channels = input_size[-1]

        self.use_ReLU = use_ReLU
        self.use_batch_normalization = use_batch_normalization

        self.dropout_p = dropout_p
        
        #Conv 1:
        #in_channels =1
        self.conv_layer_1 = ConvolutionalLayer(self.in_channels, 30, kernel_size=(3,3), kernel_stride=(1,1), ReLU = None, use_batch_normalization = False)
        self.conv_layer_2 = ConvolutionalLayer(30, 30, kernel_size=(3,3), kernel_stride=(1,1), ReLU = None, use_batch_normalization = False)
        self.max_pool1 = nn.MaxPool2d((2,2), stride=(2,2))

        #Conv 2:
        self.conv_layer_3 = ConvolutionalLayer(30, 60, kernel_size=(3,3), kernel_stride=(1,1), ReLU = None, use_batch_normalization = False)
        self.conv_layer_4 = ConvolutionalLayer(60, 60, kernel_size=(3,3), kernel_stride=(1,1), ReLU = None, use_batch_normalization = False)
        self.max_pool2 = nn.MaxPool2d((2,2), stride=(2,2))

        #Conv 3:
        self.conv_layer_5 = ConvolutionalLayer(60, 90, kernel_size=(3,3), kernel_stride=(1,1), ReLU = None, use_batch_normalization = False)
        self.conv_layer_6 = ConvolutionalLayer(90, 90, kernel_size=(3,3), kernel_stride=(1,1), ReLU = None, use_batch_normalization = False)
        self.max_pool3 = nn.MaxPool2d((2,2), stride=(2,2))

        #Conv 3:
        self.conv_layer_7 = ConvolutionalLayer(90, 120, kernel_size=(3,3), kernel_stride=(1,1), ReLU = None, use_batch_normalization = False)
        self.conv_layer_8 = ConvolutionalLayer(120, 120, kernel_size=(3,3), kernel_stride=(1,1), ReLU = None, use_batch_normalization = False)
        self.max_pool4 = nn.MaxPool2d((2,2), stride=(2,2))

        self.flatten = nn.Flatten()

        #Linear 1:
        self.dropout_1 = nn.Dropout(p=dropout_p)
        self.dense_1 = DenseLayer(1920,512)

        #Linear 2:
        self.dropout_2 = nn.Dropout(p=dropout_p)
        self.dense_2 = DenseLayer(512,10)

    def forward(self, x, debug=False):
        x = x.permute(0, 3, 1, 2) 

        # cnn layer-1
        x = self.conv_layer_1(x)
        if debug: print(x.shape)

        x = self.conv_layer_2(x)
        if debug: print(x.shape)

        x = self.max_pool1(x)
        if debug: print(x.shape)
        

        # cnn layer-2
        x = self.conv_layer_3(x)
        if debug: print(x.shape)
        x = self.conv_layer_4(x)
        if debug: print(x.shape)
        
        x = self.max_pool2(x)
        if debug: print(x.shape)
        
        # cnn layer-3
        x = self.conv_layer_5(x)
        if debug: print(x.shape)
        x = self.conv_layer_6(x)
        if debug: print(x.shape)
        
        x = self.max_pool3(x)
        if debug: print(x.shape)
        
        # cnn layer-3
        x = self.conv_layer_7(x)
        if debug: print(x.shape)
        x = self.conv_layer_8(x)
        if debug: print(x.shape)
        
        x = self.max_pool4(x)
        if debug: print(x.shape)


        #Dense
        x = self.flatten(x)
        if debug: print(x.shape)
        

        # dense layer-1
        x = self.dropout_1(x)
        if debug: print(x.shape)
        
        x = self.dense_1(x)
        if debug: print(x.shape)
        

        # dense output layer
        x = self.dropout_2(x)
        if debug: print(x.shape)
        
        x = self.dense_2(x)
        if debug: print(x.shape)
        

        return x

if __name__ == "__main__":
    """
    net = ConvolutionalNetwork(1)
    print(net)
    """