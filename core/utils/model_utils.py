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

#Taken from https://stackoverflow.com/questions/54846905/pytorch-get-all-layers-of-model
#Turn a model built by both nested nn.Sequential containers and normal layers into a flat list
def get_children(model: torch.nn.Module):
    # get children form model!
    children = list(model.children())
    flatt_children = []
    if children == []:
        # if model has no children; model is last child! :O
        return model
    else:
       # look for children from children... to the last child!
       for child in children:
            try:
                flatt_children.extend(get_children(child))
            except TypeError:
                flatt_children.append(get_children(child))
    return flatt_children