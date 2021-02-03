
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
                          tasks={"audio_classification" : "Audio classification"}, 
                          save_to_file=False, title_prefix=None, 
                          scores_on_train=False,best_epoch_bool=False):
    plt.close("all")
    scores, epoch_list, best_epoch = load_scores(model_name, model_dir, scores_on_train=scores_on_train)
    
    print("scores on : ",model_name , scores["audio_classification"]["accuracy"][best_epoch],"\n\n")
    return scores["audio_classification"]["f1"][best_epoch]

def accuracy_all_classes(model_name, model_dir, 
                          tasks={"audio_classification" : "Audio classification"}, 
                          save_to_file=False, title_prefix=None, 
                          scores_on_train=False):
    plt.close("all")
    scores, epoch_list, best_epoch = load_scores(model_name, model_dir, scores_on_train=scores_on_train)
    best_epoch_bool = False
    if best_epoch_bool is True:scores_best,epoch_list_best,best_epoch_best = load_scores(model_name,model_dir,from_epoch=best_epoch,to_epoch=best_epoch,scores_on_train=scores_on_train)
    
    print("scores on : ",model_name , scores["audio_classification"]["accuracy"][best_epoch],"\n\n")

    #print(len(scores))

    #for key, value in scores.items():
    #    print(key, ' : ', value,"\n")

    #print(scores," ",epoch_list," ",best_epoch)
    #print(scores)

    #print(epoch_list)
    #print("best epoch: ",best_epoch)
    
    plot_dir = os.path.join(model_dir,"plots","confusion_matrices")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    #print(plot_dir)
    
    for task_key, task_header in tasks.items():
        assert task_key in scores.keys(), "Scores for task "+task_key+" not found"
        
        if best_epoch_bool : 
            confusion_matrix = scores_best[task_key]["confusion matrix"]
            #print(scores_best[task_key]["confusion matrix"])
            #print(scores[task_key]["confusion matrix"])

        else:
            confusion_matrix = scores[task_key]["confusion matrix"]

      
        #print(confusion_matrix)
        len_confusion_matrix = len(confusion_matrix)
        #print(len_confusion_matrix)
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
        
        #print(sum_over_all_matrix)
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
                if i == j:
                    TPc = confusion_matrix[i][j]
                    TPs[i+1] = TPc
                    FNc = FNc - TPc
                    FNs[i+1] = FNc

        #print("sum_over_all_matrix: ",sum_over_all_matrix)
        #print("TPs: ",TPs,"\n")
        #print("FNs: ",FNs,"\n")
        sum_columns = confusion_matrix.sum(axis=0)
        #print(sum_columns)
        counter = 1
        FPs = {}
        for i in sum_columns:
            FPs[counter] = i - TPs[counter]
            counter +=1
        
        #print("FPs: ",FPs,"\n")

        TNs = {}
        for i in range(len_confusion_matrix):
            TNs[i+1] = sum_over_all_matrix - TPs[i+1] - FPs[i+1] - FNs[i+1]
            #TNs[i+1] = 0

        #print("TNs: ",TNs,"\n")

        ACCs = {}
        for i in range(len_confusion_matrix):
            Num = TPs[i+1] + TNs[i+1] 
            Den = TPs[i+1] + TNs[i+1] + FPs[i+1] + FNs[i+1]
            ACCs[i+1] = Num / Den
        
        #print("ACCs: ",ACCs)
        return ACCs

def delta_plot(data, axis_labels, x, y, show=False, save_to_dir = None):
    
    plt.close("all")

    # Create DataFrame
    df = pd.DataFrame(data)
    if len(data) == 3 : 
        hue = "augmentations"

        g = sns.catplot(
            data=df, kind="bar",
            x=x, y=y,hue = hue,
            ci="sd", palette="dark", alpha=.6, height=6
        )
        ax1 = g.axes[0]
        g.despine(left=True)
        g.set_axis_labels(axis_labels[0],axis_labels[1])
    else:
        #g = sns.catplot(
        #    data=df, kind="bar",
        #    x=x, y=y,
        #    ci="sd", palette="dark", alpha=.6, height=6, dodge = False
        #)
        plt.bar(data[x], data[y], width=0.3)


    if save_to_dir is not None:
        path = os.path.join(save_to_dir,plot_title.replace("\n", " ").replace(":"," ")+".png")
        fig.savefig(path)
    
    if show: 
        plt.show()  

    plt.close("all")

def plot_train_test_accuracy_delta(model_dir, model_names, 
                                    metrics = {"accuracy" : "Accuracy"},
                                    tasks = {"audio_classification" : "Audio classification"},
                                    show = False,
                                    save_to_dir = None,
                                    plot_title = ""
                                    ):

    deltas = {}
    for metric_name, metric_label in metrics.items():
        for task_name, task_header in tasks.items():
            deltas[task_name] = {}
            for model_name, model_plot_label in model_names.items():
                test_scores, _, best_epoch = load_scores(model_name, model_dir, scores_on_train=False)
                train_scores, _, _ = load_scores(model_name, model_dir, scores_on_train=True)
                
                assert task_name in test_scores.keys(), "Test scores for task "+task_name+" not found"
                assert task_name in train_scores.keys(), "Train scores for task "+task_name+" not found"
                
                test_metric_value = test_scores[task_name][metric_name][best_epoch]
                train_metric_value = train_scores[task_name][metric_name][best_epoch]

                deltas[task_name][model_plot_label] = train_metric_value - test_metric_value
        
            pd_data = {"augmentations": list(deltas[task_name].keys()), "deltas" : list(deltas[task_name].values())}
            delta_plot(pd_data, ("Augmentation","Train-Test {} delta".format(metric_label)), "augmentations", "deltas", show=show)


def get_accuracy_delta(accuracies_augmentation_method,accuracies_base):
        
    delta = {}
    names = ["air_conditioner","car_horn","children_playing","dog_bark","drilling","engine_idling","gun_shot","jackhammer","siren","street_music", "All classes"]
    for key,value in accuracies_base.items():
        delta[names[key-1]] = accuracies_augmentation_method[key] - value
    
    return delta   



def plot_delta_on_metric(model_dir,compute_accuracy,aug_to_test,aug_chosen_for_comparation,
                      plot_axes_labels,x,y,
                      tasks={"audio_classification" : "Audio classification"},
                      save_to_file=True, 
                      title_prefix = "Base",
                      scores_on_train=False
                      
):
    #compute all accuracies or f1_scores
    augmentation_delta_computed = {}
    for el in aug_to_test:
        if compute_accuracy:
            augmentation_delta_computed[el] = accuracy_all_classes(el, model_dir, 
                                            tasks=tasks,
                                            save_to_file=save_to_file, 
                                            title_prefix = title_prefix,
                                            scores_on_train=scores_on_train
                                            )
            
        else: 
            augmentation_delta_computed[el] = f1_score_models(el,model_dir,
                                            tasks=tasks,
                                            save_to_file=save_to_file,
                                            title_prefix=title_prefix,
                                            scores_on_train=scores_on_train
                                            )

    #compute deltas
    if compute_accuracy:
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
            deltas[key] = get_accuracy_delta(augmentation_delta_computed[key],augmentation_delta_computed[aug_chosen_for_comparation])

            for key_in,value_in in deltas[key].items():
                data[y].append(key_in)
                data[x].append(value_in)
                data[vocab[2]].append(value)
        """
        for key,value in deltas.items():
            print(key," ",value,"\n\n")
        
        for key,value in data.items():
            print(key," ",value,"\n\n")
        """
        #print(data)
        delta_plot(data,plot_axes_labels,x,y)

    
    else:
        #print(augmentation_delta_computed)
        data = {x:[],y:[]}

        for key,value in augmentation_delta_computed.items():
            if key is aug_chosen_for_comparation:
                base_value = value
            else:
                data[y].append(key)
                data[x].append(value-base_value)
        
        data_with_delta = {x:[],y:[]}
        
        #print(augmentation_delta_computed)
        
        #print(data)

        delta_plot(data,plot_axes_labels,x,y)



if __name__ == "__main__":
    model_dir = "model/test_on_fold_10"
    dict_augmentation_to_test = {
        "Base":"Base",
        "PitchShift":"PS1",
        "BackgroundNoise":"BG",
        "DynamicRangeCompression":"DRC",
        "TimeStretch":"TS"

    }
    """
    def plot_delta_on_metric(model_dir,compute_accuracy,aug_to_test,aug_chosen_for_comparation,
                      plot_axes_labels,x,y,hue,
                      tasks={"audio_classification" : "Audio classification"},
                      save_to_file=True, 
                      title_prefix = "Base",
                      scores_on_train=False
                      
    """
    #x="value", y="class", 
        #hue="augmentations",
    plot_delta_on_metric(model_dir, False, dict_augmentation_to_test, "Base", ["Δ f1-score", "augmentations"], "f1_score", "augmentations")
    plot_delta_on_metric(model_dir, True, dict_augmentation_to_test, "Base", ["Δ Classification Accuracy", "class"], "value", "class")

    #plot_delta_f1_score(model_dir)
