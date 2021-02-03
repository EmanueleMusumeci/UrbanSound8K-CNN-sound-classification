
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
from utils.plot_utils import load_scores, get_best_epoch_scores
import seaborn as sns
import pandas as pd

def f1_score_models(model_name, model_dir, 
                          scores_on_train=False):
    '''
        returns  the f1 of the model choosen
        Args:
        - model_name: directory of the model choosen
        - model_dir: directory containing all models

        OPTIONAL
        - scores_on_train
        Returns
        - returns the f1-score of the model choosen at best epoch
    ''' 
    scores, epoch_list, best_epoch = load_scores(model_name, model_dir, scores_on_train=scores_on_train)
    
    return scores["audio_classification"]["f1"][best_epoch]

def acc_score_models(model_name, model_dir, 
                          scores_on_train=False):
    '''
        returns  the accuracy of the model choosen
        Args:
        - model_name: directory of the model choosen
        - model_dir: directory containing all models

        OPTIONAL
        - scores_on_train
        Returns
        - returns the accuracy of the model choosen at best epoch
    ''' 
    scores, epoch_list, best_epoch = load_scores(model_name, model_dir, scores_on_train=scores_on_train)
    
    return scores["audio_classification"]["accuracy"][best_epoch]

def method_all_classes(model_name, model_dir, 
                          tasks={"audio_classification" : "Audio classification"}, 
                          save_to_file=False, title_prefix=None, 
                          scores_on_train=False,accuracy=True):
    '''
        Computes the accuracy for each class
        Args:
        - data: pandas data frame on which doing the plot
        - plot_axes_labes: tuple with the axis labels on the plot
        - x: var on x axis
        - y: var on y axis 
        - horizontal: True or False determines the orientation of the barplot
        - metric : title for the plot
        OPTIONAL
        - plot_dir = directory in which save the file
        Returns
        - saves in directory plots/delta plots the plot created
    ''' 
    

    plt.close("all")

    scores, epoch_list, best_epoch = load_scores(model_name, model_dir, scores_on_train=scores_on_train)
    best_epoch_bool = False
    if best_epoch_bool is True:scores_best,epoch_list_best,best_epoch_best = load_scores(model_name,model_dir,from_epoch=best_epoch,to_epoch=best_epoch,scores_on_train=scores_on_train)
    
    #print("scores on : ",model_name , scores["audio_classification"]["f1"][best_epoch],"\n\n")
    
    plot_dir = os.path.join(model_dir,"plots","confusion_matrices")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    for task_key, task_header in tasks.items():
        assert task_key in scores.keys(), "Scores for task "+task_key+" not found"
        
        if best_epoch_bool : 
            confusion_matrix = scores_best[task_key]["confusion matrix"]
        else:
            confusion_matrix = scores[task_key]["confusion matrix"]
        len_confusion_matrix = len(confusion_matrix)
        #print(len_confusion_matrix)
        #ACCURACY per classe, c rappresenta la classe
        #       TPc + TNc
        #-----------------------
        #TPc + TNc + FPc + FNc

        sum_over_all_matrix = 0
        for i in range(len_confusion_matrix):
            for j in range(len_confusion_matrix):
                sum_over_all_matrix += confusion_matrix[i][j]
        
        counter_class = 0
        TPs = {}
        FNs = {}
        for i in range(len_confusion_matrix):
            row = confusion_matrix[i]
            FNc = sum(row)
            for j in range(len_confusion_matrix):
                col = confusion_matrix[i][j]
                if i == j:
                    TPc = confusion_matrix[i][j]
                    TPs[i+1] = TPc
                    FNc = FNc - TPc
                    FNs[i+1] = FNc

        sum_columns = confusion_matrix.sum(axis=0)
        counter = 1
        FPs = {}
        for i in sum_columns:
            FPs[counter] = i - TPs[counter]
            counter +=1
        

        TNs = {}
        for i in range(len_confusion_matrix):
            TNs[i+1] = sum_over_all_matrix - TPs[i+1] - FPs[i+1] - FNs[i+1]

        ACCs = {}
        for i in range(len_confusion_matrix):
            Num = TPs[i+1] + TNs[i+1] 
            Den = TPs[i+1] + TNs[i+1] + FPs[i+1] + FNs[i+1]
            ACCs[i+1] = Num / Den
        
        return ACCs

def delta_plot(data, axis_labels, x_label, y_label, horizontal, metric, plot_dir = None):
    '''
        Creates delta plot
        Args:
        - data: pandas data frame on which doing the plot
        - plot_axes_labes: tuple with the axis labels on the plot
        - x_label: label on x axis
        - y_label: label on y axis 
        - horizontal: True or False determines the orientation of the barplot
        - metric : title for the plot
        OPTIONAL
        - plot_dir = directory in which save the file
        Returns
        - saves in directory plots/delta plots the plot created
    ''' 
    x_hori = x_label
    y_hori = y_label
    if not horizontal:
        x_hori = y_label
        y_hori = x_label
    # Create DataFrame
    df = pd.DataFrame(data)
    if len(data) == 3 : 
        hue = "augmentations"

        g = sns.catplot(
            data=df, kind="bar",
            x=x_hori, y=y_hori,hue = hue,
            ci="sd", palette="dark", alpha=.6, height=6
        )
    else:
        g = sns.catplot(
            data=df, kind="bar",
            x=x_hori, y=y_hori,
            ci="sd", palette="dark", alpha=.6, height=4
        )
    ax1 = g.axes[0]
    g.despine(left=True)
    g.set_axis_labels(axis_labels[0],axis_labels[1])
    if not horizontal:
        plt.xticks(rotation=45, ha="right") 
        plt.yticks(rotation=0) 

    
    if plot_dir is not None:
        plot_dir = os.path.join(plot_dir,"delta plots")
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)


        path = os.path.join(plot_dir,metric+"_delta_plot")+".png"
        g.savefig(path)
        
        #if plot_dir is None or show:
        plt.show()
        
    plt.close("all")

def plot_delta_on_metric(model_dir, compute, aug_to_test, aug_chosen_for_comparation, 
                      plot_axes_labels, x, y, classes_names = None,
                      tasks={"audio_classification" : "Audio classification"},
                      save_to_file=True, 
                      title_prefix = "Base",
                      scores_on_train=False,
                      horizontal = True,
                      plot_dir = None
                      
):
    '''
        Returns the delta plot chosen
        Args:
        - model_dir: directory containing all models
        - compute: which type of delta between "all" (all accuracies),"accuracy" and "f1"
        - aug_chosen_for: list of augmentation to test
        - aug_chosen_for_comparation: augmentation choosen as ground comparation
        - plot_axes_labes: tuple with the axis labels on the plot
        - x: var on x axis
        - y: var on y axis

        OPTIONAL
        - classes_names: required if compute = "all", list of classes
        - tasks
        - save_to_file
        - title_prefix
        - scores_on_train
        - horizontal: True or False determines the orientation of the barplot
        - plot_dir = directory in which save the file
        Returns
        - saves in directory plots/delta plots the plot created
    ''' 
    #compute all accuracies or f1_score total or accuracy total
    augmentation_delta_computed = {}
    for el in aug_to_test:
        if compute is "all":
            augmentation_delta_computed[el] = method_all_classes(el, model_dir, 
                                            tasks=tasks,
                                            save_to_file=save_to_file, 
                                            title_prefix = title_prefix,
                                            scores_on_train=scores_on_train,
                                            accuracy = True
                                            )
        elif compute is "f1": 
            augmentation_delta_computed[el] = f1_score_models(el, model_dir,
                                            scores_on_train=scores_on_train
                                            )
        
        else:
            augmentation_delta_computed[el] = acc_score_models(el, model_dir,
                                                            scores_on_train=scores_on_train
                                                            )

    #compute deltas
    if compute is "all":
        sums = {}
        for key,value in augmentation_delta_computed.items():
            sum_on_aug = 0
            for key_aug,value_aug in augmentation_delta_computed[key].items():
                sum_on_aug += value_aug

            sums[key] = sum_on_aug/len(augmentation_delta_computed[key])
        
        for key,value in sums.items():
            augmentation_delta_computed[key][len(augmentation_delta_computed[key])+1] = value
    
        
        data = {x:[],y:[],"augmentations":[]}
        deltas = {}
        vocab = list(data.keys())
        for key,value in aug_to_test.items():
            
            deltas[key] = {}
            #names = ["air_conditioner","car_horn","children_playing","dog_bark","drilling","engine_idling","gun_shot","jackhammer","siren","street_music", "All classes"]
            for key_in,value_in in augmentation_delta_computed[aug_chosen_for_comparation].items():
                deltas[key][classes_names[key_in-1]] = augmentation_delta_computed[key][key_in] - value_in

            for key_in,value_in in deltas[key].items():
                data[y].append(key_in)
                data[x].append(value_in)
                data[vocab[2]].append(value)
       
        delta_plot(data,plot_axes_labels,x,y,horizontal,plot_axes_labels[0],plot_dir)
    
    else:
        data = {x:[],y:[]}

        for key,value in augmentation_delta_computed.items():
            if key is aug_chosen_for_comparation:
                base_value = value
            else:
                data[y].append(key)
                data[x].append(value-base_value)
        
        delta_plot(data,plot_axes_labels,x,y,horizontal,plot_axes_labels[0], plot_dir)


if __name__ == "__main__":
    model_dir = "model/base_test_on_fold_10"
    dict_augmentation_to_test = {
        "Base":"Base",
        "PitchShift":"PS1",
        "BackgroundNoise":"BG",
        "DynamicRangeCompression":"DRC",
        "TimeStretch":"TS"

    }
    #plot delta on total accuracy
    plot_delta_on_metric(model_dir, "accuracy", dict_augmentation_to_test, 
                        "Base", ["Δ Classification Accuracy", "class"],
                        "value", "class", horizontal=False, plot_dir="plots")

    #plot delta on total f1
    plot_delta_on_metric(model_dir, "f1",dict_augmentation_to_test,
                         "Base", ["Δ f1-score", "augmentations"], 
                        "f1_score" , "augmentations",horizontal=False,plot_dir="plots")
    
    classes_names = ["air_conditioner","car_horn","children_playing","dog_bark","drilling","engine_idling","gun_shot","jackhammer","siren","street_music", "All classes"]

    #plot delta on total accuracies
    plot_delta_on_metric(model_dir, "all", dict_augmentation_to_test,
                        "Base", ["Δ Classification Accuracies", "class"],
                        "value", "class", classes_names = classes_names, horizontal=True, plot_dir="plots")
    

