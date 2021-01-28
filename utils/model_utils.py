import os

import torch
import dill


def save_model_summary(model, name, input_shape, input_channels=3, save_to_dir="models", torchsummary=False):
    if torchsummary:
        smr = summary(model, input_size=(input_channels, input_shape[0], input_shape[1]))
    else:
        smr = str(model)
        
    if not os.path.exists(save_to_dir):
        try:
            os.makedirs(save_to_dir)
        except OSError as e:
            pass
    
    with open(os.path.join(save_to_dir, name+"_summary.log"), "w") as f:
        f.write(str(smr))

def unpickle_data(path):
    with open(path, "rb") as f:
      data = dill.load(f)
      return data

def pickle_data(data, path):
    with open(path, "wb+") as f:
        dill.dump(data, f)
