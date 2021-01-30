
import os
import math

import numpy as np

import dill

from natsort import natsorted

import seaborn

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

import imageio

import torchvision
from torchvision import transforms

import pandas as pd

import PyTorchVisualizations
#from PyTorchVisualizations.src.gradcam import GradCam
#from PyTorchVisualizations.src.misc_functions import save_class_activation_images
from PyTorchVisualizations.src.misc_functions import save_class_activation_images

import librosa

import scipy

from scipy import signal

from utils.plot_utils import load_scores

def accuracy_all_classes(model_name, model_dir, 
                          tasks={"audio_classification" : "Audio classification"}, 
                          save_to_file=False, title_prefix=None, 
                          scores_on_train=False):

    plt.close("all")

    scores, epoch_list, best_epoch = load_scores(model_name, model_dir, scores_on_train=scores_on_train)

    #print(len(scores))

    #for key, value in scores.items():
    #    print(key, ' : ', value,"\n")

    #print(scores," ",epoch_list," ",best_epoch)
    #print(scores)

    #print(epoch_list)
    #print(best_epoch)
    
    plot_dir = os.path.join(model_dir,"plots","confusion_matrices")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    print(plot_dir)
    
    for task_key, task_header in tasks.items():
        assert task_key in scores.keys(), "Scores for task "+task_key+" not found"
        confusion_matrix = scores[task_key]["confusion matrix"]

        print(confusion_matrix)
        len_confusion_matrix = len(confusion_matrix)
        print(len_confusion_matrix)
        #ACCURACY per classe, c rappresenta la classe
        #       TPc + TNc
        #-----------------------
        #TPc + TNc + FPc + FNc

        """
        called on Base:

        [[72  0  6  1  0 10  0  5  2  4]    air_conditioner(and 1st column)
        [ 0 27  0  0  0  2  3  0  0  1]     car_horn
        [ 0  0 84  5  2  4  0  0  5  0]     children_playing
        [ 0  4  6 69  4  0  8  2  2  5]     dog_bark
        [ 7  3  2  0 77  1  0  3  7  0]     drilling
        [19  0  0  1  3 64  1  0  0  5]     engine_idling
        [ 0  2  0  1  0  0 29  0  0  0]     gun_shot
        [ 0  1  0  0 10  2  0 83  0  0]     jackhammer
        [ 4  0 12 13  3  3  0  0 48  0]     siren
        [ 1  2 13  0  4  0  0  0  3 77]]    street music


        c = accuracy_air_condtioner:
        (class c identified as class c)
        TPc = 72      
        (samples of classes != c but identified as class c)  
        FPc = 7+19+4+1  
        FNc = 6+1+10+5+2+4
        TNc = sum over all matrix - TPc-FPc-FNc

        """
        #sum over all matrix
        sum_over_all_matrix = 0
        for i in range(len_confusion_matrix):
            for j in range(len_confusion_matrix):
                sum_over_all_matrix += confusion_matrix[i][j]
        
        print(sum_over_all_matrix)
        counter_class = 0
        #TPs TNs FNs FPs computing
        TPs = dict()
        FNs = dict()
        for i in range(len_confusion_matrix):
            row = confusion_matrix[i]
            #print(row)
            FNc = sum(row)
            #print(FNc)
            for j in range(len_confusion_matrix):
                col = confusion_matrix[i][j]
                #FNc = sum(row)
                if i == j and j == counter_class:
                    TPc = confusion_matrix[i][j]
                    TPs[i+1] = TPc
                    counter_class += 1
                    FNc = FNc - TPc
                    FNs[i+1] = FNc

        print("sum_over_all_matrix: ",sum_over_all_matrix)
        print("TPs: ",TPs,"\n")
        print("FNs: ",FNs,"\n")
        sum_columns = confusion_matrix.sum(axis=0)
        #print(sum_columns)
        counter = 1
        FPs = dict()
        for i in sum_columns:
            FPs[counter] = i - TPs[counter]
            counter +=1
        
        print("FPs: ",FPs,"\n")

        TNs = dict()

        for i in range(len_confusion_matrix):
            TNs[i+1] = sum_over_all_matrix - TPs[i+1] - FPs[i+1] - FNs[i+1]

        print("TNs: ",TNs,"\n")

        ACCs =dict()

        for i in range(len_confusion_matrix):
            Num = TPs[i+1] + TNs[i+1] 
            Den = TPs[i+1] + TNs[i+1] + FPs[i+1] + FNs[i+1]
            ACCs[i+1] = Num / Den
        
        print("ACCs: ",ACCs)
        return ACCs


def delta_accuracy_plot(data):
    import seaborn as sns
    import pandas as pd
    
    # Create DataFrame
    df = pd.DataFrame(data)
 
    g = sns.catplot(
        data=df, kind="bar",
        x="value", y="class", hue="augmentations",
        ci="sd", palette="dark", alpha=.6, height=6
    )
    g.despine(left=True)
    g.set_axis_labels("Δ Classification Accuracy", "class")
    g.legend.set_title("")

    plt.show()

def update_data(data,delta,augmentation):

    for key,value in delta.items():
        data["class"].append(key)
        data["value"].append(value)
        data["augmentations"].append(augmentation)
    return data    

def get_accuracy_delta(accuracies_augmentation_method,accuracies_base):
        
    delta = {}
    names = ["air_conditioner","car_horn","children_playing","dog_bark","drilling","engine_idling","gun_shot","jackhammer","siren","street_music"]
    for key,value in accuracies_base.items():
        delta[names[key-1]] = accuracies_augmentation_method[key] - value
    
    return delta   
  
                

if __name__ == "__main__":
    model_dir = "model"
    accuracies_base = accuracy_all_classes("Base", model_dir, 
                                        tasks={"audio_classification":"Audio classification"},
                                        save_to_file=True, 
                                        title_prefix = "Base",
                                        scores_on_train=False
                                        )
    
    accuracies_PS1 = accuracy_all_classes("PitchShift", model_dir, 
                                        tasks={"audio_classification":"Audio classification"},
                                        save_to_file=True, 
                                        title_prefix = "Base",
                                        scores_on_train=False
                                        )
    
    accuracies_PS2 = accuracy_all_classes("PitchShift_PS2", model_dir, 
                                        tasks={"audio_classification":"Audio classification"},
                                        save_to_file=True, 
                                        title_prefix = "Base",
                                        scores_on_train=False
                                        )
    accuracies_BG = accuracy_all_classes("BackgroundNoise", model_dir, 
                                        tasks={"audio_classification":"Audio classification"},
                                        save_to_file=True, 
                                        title_prefix = "Base",
                                        scores_on_train=False
                                        )
    
    accuracies_DRC = accuracy_all_classes("DynamicRangeCompression", model_dir, 
                                        tasks={"audio_classification":"Audio classification"},
                                        save_to_file=True, 
                                        title_prefix = "Base",
                                        scores_on_train=False
                                        )

    accuracies_TS = accuracy_all_classes("TimeStretch", model_dir, 
                                        tasks={"audio_classification":"Audio classification"},
                                        save_to_file=True, 
                                        title_prefix = "Base",
                                        scores_on_train=False
                                        )
    
    
    data = {"class":[],"value":[],"augmentations":[]}

    TS_delta = get_accuracy_delta(accuracies_TS,accuracies_base)
    print("TS_delta: ",TS_delta)
    data = update_data(data,TS_delta,"TS")
    print(data)

    PS1_delta = get_accuracy_delta(accuracies_PS1,accuracies_base)
    print("PS1_delta: ",PS1_delta)
    data = update_data(data,PS1_delta,"PS1")
    print(data)
    
    PS2_delta = get_accuracy_delta(accuracies_PS2,accuracies_base)
    print("PS2_delta: ",PS2_delta)
    data = update_data(data,PS1_delta,"PS2")
    print(data)

    DRC_delta = get_accuracy_delta(accuracies_DRC,accuracies_base)
    print("DRC_delta: ",DRC_delta)
    data = update_data(data,DRC_delta,"DRC")
    print(data)

    BG_delta = get_accuracy_delta(accuracies_BG,accuracies_base)
    print("BG_delta: ",BG_delta)
    data = update_data(data,BG_delta,"BG")
    print(data)
    
   

    delta_accuracy_plot(data)
