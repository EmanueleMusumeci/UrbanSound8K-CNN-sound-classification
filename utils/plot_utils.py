
import os
import math

import numpy as np

import dill

from natsort import natsorted

import seaborn

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

import pandas as pd

import imageio

import torchvision
from torchvision import transforms

import PyTorchVisualizations
from PyTorchVisualizations.src.gradcam import GradCam
from PyTorchVisualizations.src.misc_functions import save_class_activation_images

import librosa.display

import scipy

from scipy import signal

from utils.model_utils import get_children

def load_scores(model_name, model_dir, 
                from_epoch=0, to_epoch=0, epochs_skip=0, 
                scores_subdir=None, 
                scores_on_train=False):

  scores_directory = os.path.join(model_dir,model_name)

  if scores_subdir is not None:
    scores_directory = os.path.join(scores_directory,scores_subdir)
  else:
    if os.path.exists(os.path.join(scores_directory,"scores_on_test")) and not scores_on_train:
      scores_directory = os.path.join(scores_directory,"scores_on_test")
    else:
      scores_directory = os.path.join(scores_directory,"scores_on_train")

  scores = {}
  epochs_list = []
  
  best_epoch = 0 #Find best epoch based on f1 score
  best_f1 = 0

  #read files in alphabetical order
  for filename in natsorted(os.listdir(scores_directory)):
    if not os.path.isfile(os.path.join(scores_directory,filename)):
      continue
    elif filename.endswith(".scores"):
      scores_path = os.path.join(scores_directory, filename)
      with open(scores_path, "rb") as f:
        scores_entry = dill.load(f)

        if scores_entry["Epoch"] < from_epoch:
          continue
        elif to_epoch>0 and scores_entry["Epoch"] > to_epoch:
          break
        else:
          if epochs_skip>0 and (scores_entry["Epoch"]-from_epoch)%epochs_skip!=0:
            continue
        epochs_list.append(scores_entry["Epoch"])
        for key, value in scores_entry.items():
          if key=="Epoch": continue
          else:
            if key not in scores.keys():
              scores[key] = {}
            if scores_entry[key] is None: continue
            for score_name, score_value in scores_entry[key].items():
              if score_name=="confusion matrix":
                  if scores_entry[key]["f1"] >= best_f1:
                    best_epoch = scores_entry["Epoch"]
                    best_f1 = scores_entry[key]["f1"]
                    scores[key]["confusion matrix"] = score_value
              elif score_name=="distribution":
                  if scores_entry[key]["f1"] >= best_f1:
                    best_epoch = scores_entry["Epoch"]
                    best_f1 = scores_entry[key]["f1"]
                    scores[key]["distribution"] = score_value
              else:
                if score_name not in scores[key].keys():
                  scores[key][score_name] = {}
                scores[key][score_name][scores_entry["Epoch"]] = scores_entry[key][score_name]
  
  return scores, epochs_list, best_epoch

###########
#  PLOTS  #
###########
'''
Utils to generate all plots and graphical renders needed for the presentation
'''
def plot_scores(model_name, model_dir, tasks={"audio_classification" : "Audio classification"},
                metrics={"F1-macro":["f1"]},
                from_epoch=0, to_epoch=0, epochs_skip=0,
                xticks_step=0, combine_tasks=False, increase_epoch_labels_by_one=False, 
                title_prefix=None, color = None, scores_on_train=False,
                plot_dir = None, show=False):
  '''
  Plots the requested performance metrics from the score history of the current model
  (used to generate the graphs in the report)
  Args:
    - model_name: name of the model that generated the scores to be plotted
    - model_dir: directory containing the models
    OPTIONAL
    - tasks: dictionary containing tasks whose score is going to be plotted and
             the name to print on the plot
    - from_epoch: plot scores starting from a certain episode (default: 0)
    - to_epoch: plot scores starting until a certain episode (default: 0, plot all scores)
    - epochs_skip: allows skipping n epochs every plotted point (if scores where saved
                   every n epochs) (default: 0)
    - xticks_step: used to print a "tick" on the graph x axis every n ticks 
      (with a value of 5 we would have 0,5,10... on the x axis)
    - combine_tasks: allows plotting scores for all tasks on the same plot (False)
  '''

  plt.close("all")

  assert plot_dir is not None or show, "No directory for saving the plot has been specified and the \
                                        plot is not being shown. No point in making a plot :)"

  #1) Load scores
  scores, epochs_list, best_epoch = load_scores(model_name, model_dir, 
                                    from_epoch=from_epoch, to_epoch=to_epoch,
                                    epochs_skip=0, scores_on_train=scores_on_train)
  
  #2) Plot requested scores
  plots = {}
  seaborn.reset_orig()
  for task_key, task_header in tasks.items():
    assert task_key in scores.keys(), "Scores for task "+task_key+" not found"
    #metric_keys will contains all metrics we want to appear together in the
    #combined plot
    for plot_header, metric_keys in metrics.items():
      if combine_tasks:
        if plot_header in plots.keys():
          current_plot = plt.figure(model_name+"_"+plot_header)
        else:
          current_plot = plt.figure(plot_header)
          plots[plot_header] = current_plot
          
        if title_prefix is None:
            title = "Combined tasks\n"+plot_header
        else:
            title = title_prefix+" - Combined tasks\n"+plot_header
      else:
        current_plot = plt.figure(model_name+"_"+task_header+"_"+plot_header)
        plots[task_header+"_"+plot_header] = current_plot
        if title_prefix is None:
            title=task_header+"\n"+plot_header
        else:
            title = title_prefix+"\n"+plot_header

      plt.title(title)

      if increase_epoch_labels_by_one:  
        labels_list = [epoch+1 for epoch in epochs_list[::xticks_step]]
      else:
        labels_list = epochs_list[::xticks_step]

      plt.xticks(ticks=range(0,len(epochs_list),xticks_step), labels=labels_list)
      plt.xlabel("Epoch")
      for metric in metric_keys:
        #2.1) Collect scores for this metric in a list
        values = []
        for _, value in sorted(scores[task_key][metric].items()):
          values.append(value)
        if combine_tasks:
          label=task_header
        else:
          label=metric.capitalize()
        if color is not None:
            plt.plot(values, label=label, color = color)
        else:
            plt.plot(values, label=label)

        if len(metric_keys)>1 or combine_tasks:
          plt.legend(loc="lower right")
        else:
          plt.ylabel(metric.capitalize())
  
  if plot_dir is not None:
    plot_dir = os.path.join(plot_dir, model_name)
    if not os.path.exists(plot_dir):
      os.makedirs(plot_dir)
    for k,plot in plots.items():
      path = os.path.join(plot_dir, model_name+" - "+k)+".png"
      plot.savefig(path)
  
  if plot_dir is None or show:
    for k,plot in plots.items():
      plot.show()

  
def plot_confusion_matrix(model_name, model_dir, 
                          tasks={"audio_classification" : "Audio classification"}, 
                          title_prefix=None, scores_on_train=False, 
                          plot_dir = None, show = False):

  plt.close("all")

  assert plot_dir is not None or show, "No directory for saving the plot has been specified and the \
                                        plot is not being shown. No point in making a plot :)"

  scores, epoch_list, best_epoch = load_scores(model_name, model_dir, scores_on_train=scores_on_train)
  

  for task_key, task_header in tasks.items():
    assert task_key in scores.keys(), "Scores for task "+task_key+" not found"
    confusion_matrix = scores[task_key]["confusion matrix"]
    current_plot = plt.figure(task_header+"_Confusion_matrix")
    
    #Normalize the confusion matrix to broaden color ranges
    data = np.array(confusion_matrix)
    max_val = max(map(max,confusion_matrix))

    #the sklearn confusion matrix function uses labels in lexicographic order!
    labels = sorted([key for key, _ in scores[task_key]["distribution"].items()])

    #use the max between the actual min of the confusion matrix and 1 to 
    #avoid problems with the non linear scale
    min_val = max(1,min(map(min,confusion_matrix)))
    #then replace all zeros with this values
    for i,row in enumerate(confusion_matrix):
      for j,value in enumerate(row):
        if value==0:
          confusion_matrix[i][j] = min_val

    avg_val = sum(sum(confusion_matrix))/(len(confusion_matrix)*len(confusion_matrix[0]))
    fig, axes = plt.subplots(figsize=(8,8))

    #seaborn.set(font_scale=2)
    # TODO Michele Confusion matrix , discalie oblique (no orizzontali)
    ax = seaborn.heatmap(confusion_matrix, 
                    norm = LogNorm(vmin=min_val, vmax=max_val),
                    cbar_kws={"shrink": 0.5, "ticks":[0,1,10,1e2,1e3,1e4,1e5]},#"orientation": "oblique"}, 
                    annot=True, ax = axes, fmt='g',
                    xticklabels=True
                    ) #annot=True to annotate cells, fmt='g' to avoid scientific notation
    ax.figure.subplots_adjust(left=0.2, bottom=0.2)

    plt.xticks(rotation=45, ha="right") 
    plt.yticks(rotation=0) 
    #heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=30) 
    #ax.set_xticklabels(ax.get_xticklabels(), rotation=10) 

    # labels, title and ticks
    axes.set_xlabel('Predicted labels', size = 10)
    axes.set_ylabel('True labels', size = 10) 
    axes.xaxis.set_ticklabels(labels, size = 10)
    axes.yaxis.set_ticklabels(labels, size = 10)
    
    if title_prefix is not None:
        conf_matrix_title = title_prefix
    else:
        conf_matrix_title = model_name
    axes.set_title(conf_matrix_title+"\nEpoch: "+str(best_epoch)+'\nConfusion Matrix') 


  if plot_dir is not None:
    plot_dir = os.path.join(plot_dir, model_name, "confusion_matrices")
    if not os.path.exists(plot_dir):
      os.makedirs(plot_dir)

    path = os.path.join(plot_dir, model_name+"_Confusion_matrix")+".png"
    fig.savefig(path)
    
    if plot_dir is None or show:
      plt.show()
      
  plt.close("all")
      
def plot_scores_from_multiple_dirs(
                model_name, model_dir, score_dirs, tasks=None,
                metrics={"F1-macro":["f1"]},
                from_epoch=0, to_epoch=0, epochs_skip=0,
                xticks_step=0, combine_tasks=False, increase_epoch_labels_by_one=False, 
                title_prefix=None, colors = None, switch_labels = False,
                plot_dir = None, show=False):
  '''
  Plots the requested performance metrics from the score history of the current model
  (used to generate the graphs in the report)
  Args:
    - model_name: name of the model that generated the scores to be plotted
    - model_dir: directory containing the models
    OPTIONAL
    - tasks: dictionary containing tasks whose score is going to be plotted and
             the name to print on the plot
    - from_epoch: plot scores starting from a certain episode (default: 0)
    - to_epoch: plot scores starting until a certain episode (default: 0, plot all scores)
    - epochs_skip: allows skipping n epochs every plotted point (if scores where saved
                   every n epochs) (default: 0)
    - xticks_step: used to print a "tick" on the graph x axis every n ticks 
      (with a value of 5 we would have 0,5,10... on the x axis)
    - combine_tasks: allows plotting scores for all tasks on the same plot (False)
  '''

  plt.close("all")

  assert plot_dir is not None or show, "No directory for saving the plot has been specified and the \
                                        plot is not being shown. No point in making a plot :)"

  #1) Load scores

  all_scores = {}
  epochs_list = []
    
  for scores_header, scores_subdir in score_dirs.items():
    scores, epochs_list, _ = load_scores(model_name, model_dir, 
                                        from_epoch=from_epoch, to_epoch=to_epoch,
                                        epochs_skip=epochs_skip,  scores_subdir=scores_subdir)

    all_scores[scores_header] = scores

  #2) Plot requested scores
  plots = {}
  seaborn.reset_orig()
  for scores_header, scores in all_scores.items():
      for task_key, task_header in tasks.items():
          assert task_key in scores.keys(), "Scores for task "+task_key+" not found"
          #metric_keys will contains all metrics we want to appear together in the
          #combined plot
          for plot_header, metric_keys in metrics.items():
            if combine_tasks:
              if plot_header in plots.keys():
                current_plot = plt.figure(model_name+"_"+plot_header)
              else:
                current_plot = plt.figure(model_name+"_"+plot_header)
                plots[plot_header] = current_plot
                
              if title_prefix is None:
                  title = "Combined tasks\n"+"Train/Test "+plot_header
              else:
                  title = title_prefix+" - Combined tasks\n"+"Train/Test "+plot_header
            else:
              current_plot = plt.figure(model_name+"_"+task_header+"_"+plot_header)
              plots[task_header+"_"+plot_header] = current_plot
              if title_prefix is None:
                  title=task_header+"\n"+"Train/Test "+plot_header
              else:
                  title = title_prefix+"\n"+"Train/Test "+plot_header
      
            plt.title(title)
      
            if increase_epoch_labels_by_one:  
              labels_list = [epoch+1 for epoch in epochs_list[::xticks_step]]
            else:
              labels_list = epochs_list[::xticks_step]
      
            plt.xticks(ticks=range(0,len(epochs_list),xticks_step), labels=labels_list)
            plt.xlabel("Epoch")
            for metric in metric_keys:
              #2.1) Collect scores for this metric in a list
              values = []
              for _, value in sorted(scores[task_key][metric].items()):
                values.append(value)
              
              if switch_labels:
                if scores_header == "Train":
                  label = "Test"
                elif scores_header == "Test":
                  label = "Train"
              else:
                label=scores_header
              color = colors[label]

              if color is not None:
                  plt.plot(values, label=label, color = color)
              else:
                  plt.plot(values, label=label)
      
              plt.legend(loc="lower right")
  
  if plot_dir is not None:
    plot_dir = os.path.join(plot_dir, model_name)
    if not os.path.exists(plot_dir):
      os.makedirs(plot_dir)
    for k,plot in plots.items():
      path = os.path.join(plot_dir, model_name+" - Train_Test comparison - "+k)+".png"
      plot.savefig(path)

  if plot_dir is None or show:
    for k,plot in plots.items():
      plot.show()
    
def comparative_plots(model_names, model_dir, 
                      tasks={"audio_classification" : "Audio classification"},
                      metrics={"F1-macro":["f1"]}, 
                      from_epoch=0, to_epoch=0, epochs_skip=0,
                      xticks_step=0, increase_epoch_labels_by_one=False, scores_on_train=False,
                      title_prefix=None,
                      plot_dir = None, show = False,
                      ):
  '''
  Plots the requested performance metrics from the score history of the specified models
  all on the same plot (for a model comparison plot)
  (used to generate the graphs in the report)
  Args:
    - model_names: list of the names of models to be plotted
    - model_dir: directory containing the models
    OPTIONAL
    - tasks: dictionary containing tasks whose score is going to be plotted and
             the name to print on the plot
    - from_epoch: plot scores starting from a certain episode (default: 0)
    - to_epoch: plot scores starting until a certain episode (default: 0, plot all scores)
    - epochs_skip: allows skipping n epochs every plotted point (if scores where saved
                   every n epochs) (default: 0)
    - xticks_step: used to print a "tick" on the graph x axis every n ticks 
      (with a value of 5 we would have 0,5,10... on the x axis)
  '''

  plt.close("all")

  assert plot_dir is not None or show, "No directory for saving the plot has been specified and the \
                                        plot is not being shown. No point in making a plot :)"

  #1) Load scores for each model
  all_scores = {}

  #Epochs list is going to hold the available epochs in the interval [from_epoch, to_epoch]
  #for the model with most epochs (the one that has been trained longest)
  epochs_list = []

  #1) Load scores
  for model_name, model_header in model_names.items():
    scores, epochs_list, _ = load_scores(model_name, model_dir, 
                                          from_epoch=from_epoch, to_epoch=to_epoch,
                                          epochs_skip=epochs_skip, scores_on_train=scores_on_train)
    all_scores[model_name] = scores

  names = [model_header for _, model_header in model_names.items()]
  names = ", ".join(names)

  #2) Plot requested scores
  plots = {}
  seaborn.reset_orig()
  for task_key, task_header in tasks.items():
    #print(task_key)
    for plot_header, metric_keys in metrics.items():
      current_plot = plt.figure("Combined ("+names+") - "+task_header+"_"+plot_header)
      if title_prefix is None:
        title="Combined plots\n"+task_header+"\n"+plot_header
      else:
        title=title_prefix+"\n"+task_header+"\n"+plot_header

      plt.title(title)
      for model_name, model_scores in all_scores.items():

        assert task_key in model_scores.keys(), "Scores for model "+model_names[model_name] + " for task "+task_key+" not found"

        if increase_epoch_labels_by_one:  
            labels_list = [epoch+1 for epoch in epochs_list[::xticks_step]]
        else:
            labels_list = epochs_list[::xticks_step]

        plt.xticks(ticks=range(0,len(epochs_list),xticks_step), labels=labels_list)
        plt.xlabel("Epoch")
        for metric in metric_keys:
          #2.1) Collect scores for this metric in a list
          values = []
          if len(model_scores[task_key])==0: continue
          #print(model_scores[task_key][metric])
          for _, value in sorted(model_scores[task_key][metric].items()):
            values.append(value)
          label=model_names[model_name]
          plt.plot(values, label=label)
          plt.legend(loc="lower right")
          plt.ylabel(metric.capitalize())
      plots[task_header+"_"+plot_header] = current_plot
    
  if plot_dir is not None:
    plot_dir = os.path.join(plot_dir, "comparative plots")
    if not os.path.exists(plot_dir):
      os.makedirs(plot_dir)
    for k,plot in plots.items():
      path = os.path.join(plot_dir, title_prefix+" - "+k)+".png"
      plot.savefig(path)
  
  if plot_dir is None or show:
    for k,plot in plots.items():
      plot.show()

  plt.close("all")

def get_best_epoch_scores(model_name, model_dir, 
                          tasks={"audio_classification" : "Audio classification"},
                          metrics={"accuracy":"Accuracy"}, print_scores=True):
    
    #1) Load scores
    scores, epochs_list, best_epoch = load_scores(model_name, model_dir)

    best_scores = {}
    best_scores_str = "-"*40 + "\n" + model_name + "\n" + "-"*40 + "\n"
    best_scores_str += "Epoch: "+str(best_epoch)+"\n"

    for task_name in tasks.keys():
      for score_name, values in scores[task_name].items():
          if score_name in metrics.keys():
              if score_name == "confusion matrix" or score_name == "distribution":
                  best_scores_str+=str(metrics[score_name]) + ":\n"+str(values)+"\n"
                  best_scores[score_name] = values
              else:
                  best_scores[score_name] = values[best_epoch]
                  best_scores_str+=str(metrics[score_name]) + ": "+str(values[best_epoch])+"\n"

    best_scores_str += "\n\n"
    if print_scores:
      print(best_scores_str)
    return best_scores, best_scores_str

#Taken from https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/8
def plot_grad_flow(gradient_magnitudes, epoch, show=False, save_to_dir=None):

  layers = []
  for layer_name, _ in gradient_magnitudes.items():
    layers.append(layer_name.split(".")[0])
  max_grads = [entry["max_grads"] for layer_name, entry in gradient_magnitudes.items()]
  #min_grads = [entry["min_grad"] for layer_name, entry in gradient_magnitudes.items()]
  avg_grads = [entry["avg_grads"] for layer_name, entry in gradient_magnitudes.items()]

  plt.rc('xtick',labelsize=15)
  plt.rc('ytick',labelsize=15)
  plt.figure(figsize=(10,10))
  
  plt.hlines(0, 0, len(avg_grads)+1, lw=2, color="k" )
  plt.xticks(range(0,len(avg_grads), 1), layers, rotation=45, ha="right")
  
  plt.xlim(left=-1, right=len(avg_grads))
  plt.ylim(bottom = 0, top=0.1) # zoom in on the lower gradient regions
  plt.xlabel("Layers", size = 15)
  plt.ylabel("Gradient magnitude", size = 17)

  plt.subplots_adjust(left=0.2, bottom=0.2)

  plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.5, lw=1, color="c", align="center")
  plt.bar(np.arange(len(max_grads)), avg_grads, alpha=0.5, lw=1, color="b", align="center")

  title = "Gradient flow"
  if epoch is not None:
    title += " - Epoch: "+str(epoch)
  plt.title(title, fontsize=15)
  plt.grid(True)
  plt.legend([matplotlib.lines.Line2D([0], [0], color="c", lw=4),
              matplotlib.lines.Line2D([0], [0], color="b", lw=4),
              matplotlib.lines.Line2D([0], [0], color="k", lw=4)], ['Max. gradient', 'Avg. gradient', 'Zero gradient'], 
              fontsize = 15, loc = "upper right")
  
  if show:
      plt.show()

  if save_to_dir is not None:
    path = os.path.join(save_to_dir,"Gradient flow - Epoch "+str(epoch))+".png"
    plt.savefig(path)
  
  plt.close("all")

def create_gradient_flow_gif(model_name, model_dir, plot_dir,
                              tasks={"audio_classification" : "Audio classification"},
                              cropX = None, cropY = None, frame_duration = 0.3):

  scores, epochs_list, best_epoch = load_scores(model_name, model_dir)
  
  assert isinstance(plot_dir, str), "Please specify a directory for storing plots"
  
  plot_dir = os.path.join(plot_dir, model_name, "gradient_flow")  
  if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

  for task_name, task_header in tasks.items():
    gradient_images = []
    for epoch in epochs_list:
      plot_grad_flow(scores[task_name]["gradient_stats"][epoch], epoch, show=False, save_to_dir=plot_dir)
  
  for filename in natsorted(os.listdir(plot_dir)):
    if not filename.endswith(".png") or not os.path.isfile(os.path.join(plot_dir,filename)):
      continue
    #print(os.path.join(plot_dir,filename))
    image = imageio.imread(os.path.join(plot_dir, filename))
    if cropX is not None:
      image = image[cropX[0]:cropX[1],:,:]
    if cropY is not None:
      image = image[:,cropY[0]:cropY[1],:]
    #plt.imshow(image)
    gradient_images.append(image)
  imageio.mimsave(os.path.join(plot_dir, model_name+"_Gradient_flow.gif"), gradient_images, duration = frame_duration)

def show_image_preprocessing(transformations, image, title_prefix="", progressive=True, save_to_dir=None):
    images = {}
    images["0 - Original"] = plt.figure("0 - Original")
    plt.imshow(np.array(image))
    plt.axis("off")

    if progressive:
        progressive_transformations = []
        for transformation_name, transformation in transformations.items():
            progressive_transformations.append(transformation)
            pipeline = transforms.Compose(progressive_transformations)
            preprocessed_image = pipeline(image)
            images[transformation_name] = plt.figure(transformation_name)
            plt.imshow(preprocessed_image)
            plt.axis("off")
    else:
        pipeline = transforms.Compose(transformations)
        preprocessed_image = pipeline(image)
        images["Complete pipeline"] = plt.figure("Complete pipeline")
        plt.imshow(preprocessed_image)
        plt.axis("off")

    if save_to_dir is not None:
        if not os.path.exists(save_to_dir):
            os.makedirs(save_to_dir)
    for transformation_name, image in images.items():
        print(transformation_name)
        #plot.show()
        path = os.path.join(save_to_dir,title_prefix+" - Preprocessed image - "+transformation_name)+".png"
        image.savefig(path)

def plot_class_distributions(distributions, plot_dir=None):
    plt.rc('xtick',labelsize=15)
    plt.rc('ytick',labelsize=15)

    #fig = plt.figure(figsize=(18,7))
    fig, ax = plt.subplots(figsize=(18,7))
    seaborn.barplot(data = pd.DataFrame(distributions, index=[0]).melt(), 
                    x = "variable", y="value", hue="variable").legend_.remove()
    
    #plt.hlines(0, 0, len(distributions)+1, lw=2, color="k" )
    plt.xticks(range(0,len(distributions), 1), distributions.keys(), rotation=45, ha="right")
    
    plt.xlabel("Class name", size = 15)
    plt.ylabel("Instance count", size = 17)

    plt.subplots_adjust(left=0.2, bottom=0.3)

    title = "Class distribution"
    
    plt.title(title, fontsize=15)

    plt.grid(True)

    if plot_dir is not None:
        fig.savefig(os.path.join(plot_dir,"class_distribution_plot.png"))
    else:
        fig.show()
        
def visualize_features(model, convolutional_layers, dense_layers, sample, sample_label_idx, save_to_dir, filename=None, target_layer=0):
  if not os.path.exists(save_to_dir):
    os.makedirs(save_to_dir)

  # Grad cam
  grad_cam = GradCam(model.cuda(), convolutional_layers, dense_layers, target_layer=target_layer)
  # Generate cam mask 
  cam = grad_cam.generate_cam(sample["preprocessed_spectrogram"].permute(0, 3, 1, 2).cuda(), sample_label_idx)
  if filename is None:
    filename = type(model).__name__
  # Save mask
  transposed_image = np.transpose(sample["preprocessed_spectrogram"][0][:,:,0].cpu().numpy())
  save_class_activation_images(transposed_image, cam, filename, directory=save_to_dir, layer = target_layer)
  print('Grad cam completed')

def create_gif(directory, file_startswith, file_endswith="", gif_name="sequence", save_to_dir=None):

  filenames = []
  for filename in natsorted(os.listdir(directory)):
      if filename.startswith(file_startswith) and filename.endswith(file_endswith):
          filenames.append(os.path.join(directory, filename))

  images = []
  for filename in filenames:
      images.append(imageio.imread(filename))

  if save_to_dir is None:
    save_to_dir = directory
  imageio.mimsave(os.path.join(save_to_dir,gif_name)+".gif", images, duration = 0.5)

def visualize_features_on_layers(model, convolutional_layers, dense_layers,
                                image, image_label_idx, 
                                save_to_dir,
                                from_layer=0, to_layer=0,
                                filename_prefix=None, make_gif=True):
  assert from_layer<=to_layer, "Impossible range: ("+str(from_layer)+", "+str(to_layer)+")"
  
  children = get_children(model)
  layers = {}
  for i, layer in enumerate(children):
    layers[i] = layer

  #Flatten model into a list of layers with get_children
  if not os.path.exists(save_to_dir):
    os.makedirs(save_to_dir)

  if to_layer==0:
    to_layer = len(layers)-1
  for layer in range(from_layer, to_layer):
    #try:
    print("Visualizing features on layer {}".format(layer))
    visualize_features(model, convolutional_layers, dense_layers, image, image_label_idx, save_to_dir, filename=filename_prefix+str(layer), target_layer=layer)
    #except Exception as e:
    #  print(e)
    #  pass
  
  if make_gif:
    create_gif(save_to_dir, "", file_endswith="On_Image.png", gif_name="Layer_activations", save_to_dir = save_to_dir)


'''
Displays a wave plot for the input raw sound (using the Librosa library)
'''
def plot_sound_waves(sound, compare_with_sound = None, preprocessing_name = None, preprocessing_value = None,
                      horizontal = False,
                      show=False, save_to_dir=None,
                      sr=22050, plot_title=""):
  plt.close("all")

  if compare_with_sound is None:
    horizontal = False

  plot_title += "Wave plot"
  if compare_with_sound is not None:
    plot_title += "\n"+preprocessing_name

  if compare_with_sound is not None:
    plot_title += "s"
    if horizontal:
      plt.subplots(1,2, figsize=(14,5))
    else:
      plt.subplots(2,1)
  else:
    plt.subplots(1)
    
  plt.suptitle(plot_title)  

  if compare_with_sound is not None:
    if horizontal:
      plt.subplot(1,2,1)
    else:
      plt.subplot(2,1,1)
    plt.title("Original")

  librosa.display.waveplot(np.array(sound),sr=sr, x_axis="time")
  
  #plt.ylim([1e-7, 1e2])
  plt.ylabel('Magnitude (norm)')

  if compare_with_sound is None or horizontal:
    plt.xlabel('Time [sec]')
  else:
    plt.xlabel('')
  
  if compare_with_sound is not None:
    if horizontal:
      plt.subplot(1,2,2)
    else:
      plt.subplot(2,1,2)

    subplot_title = "Preprocessed"
    if preprocessing_name is not None:
      subplot_title += " ("+preprocessing_name+")"
    plt.title(subplot_title)
    
    librosa.display.waveplot(np.array(compare_with_sound),sr=sr, x_axis = "time")

    #plt.ylim([1e-7, 1e2])
    plt.xlabel('Time [sec]')
    plt.ylabel('Magnitude (norm)')
    plt.subplots_adjust(hspace= 0.4)
  
  if show: 
    plt.show()
  
  if save_to_dir is not None:
    path = os.path.join(save_to_dir,plot_title.replace("\n", " ")+".png")
    plt.savefig(path)

def plot_sound_spectrogram(sound, compare_with_sound = None, preprocessing_name = None, preprocessing_value = None,
                            horizontal = False,
                            sr=22050, log_scale = False, hop_length=512, 
                            show = False, save_to_dir = None, 
                            colorbar_format = "%+2.f dB", plot_title=""):
  plt.close("all")
  
  if compare_with_sound is None:
    horizontal = False

  plot_title += "Spectrogram"
  if compare_with_sound is not None:
    plot_title += "\n"+preprocessing_name
  
  
  if log_scale:
      y_axis = "log"
  else:
      y_axis = "linear"

  sound = librosa.stft(sound, hop_length = hop_length)
  sound = librosa.amplitude_to_db(np.abs(sound), ref=np.max)

  if compare_with_sound is not None:
    plot_title += "s"
    if horizontal:
      plt.subplots(1,2, figsize=(14,5))
    else:
      plt.subplots(2,1)
  else:
    plt.subplots(1)
    
  plt.suptitle(plot_title)  

  if compare_with_sound is not None:
    if horizontal:
      plt.subplot(1,2,1)
    else:
      plt.subplot(2,1,1)
    plt.title("Original")

  librosa.display.specshow(sound, hop_length = hop_length, x_axis="time", y_axis=y_axis)
  #if not horizontal:
  #  plt.colorbar(format=colorbar_format)
  
  #plt.ylim([1e-7, 1e2])
  plt.ylabel('Magnitude (norm)')

  if compare_with_sound is None or horizontal:
    plt.xlabel('Time [sec]')
  else:
    plt.xlabel('')

  if compare_with_sound is not None:
    compare_with_sound = librosa.stft(compare_with_sound, hop_length = hop_length)
    compare_with_sound = librosa.amplitude_to_db(np.abs(compare_with_sound), ref=np.max)

    if horizontal:
      plt.subplot(1,2,2)
    else:
      plt.subplot(2,1,2)

    subplot_title = "Preprocessed"
    if preprocessing_name is not None:
      subplot_title += " ("+preprocessing_name+")"
    plt.title(subplot_title)
    
    librosa.display.specshow(sound, hop_length = hop_length, x_axis="time", y_axis=y_axis)
    #plt.colorbar(format=colorbar_format)

    #plt.ylim([1e-7, 1e2])
    plt.xlabel('Time [sec]')
    plt.ylabel('Magnitude (norm)')
    plt.subplots_adjust(hspace= 0.4)
  
  if show: 
    plt.show()
  
  if save_to_dir is not None:
    path = os.path.join(save_to_dir,plot_title.replace("\n", " ")+".png")
    plt.savefig(path)

#from https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.periodogram.html
def plot_periodogram(sound, compare_with_sound = None, preprocessing_name = None, preprocessing_value = None,
                      horizontal = False,
                      sr=22050, plot_title="",
                      show = False, save_to_dir = None):
  plt.close("all")
  
  if compare_with_sound is None:
    horizontal = False

  plot_title += "Periodogram"
  if compare_with_sound is not None:
    plot_title += "\n"+preprocessing_name
  

  if compare_with_sound is not None:
    plot_title += "s"
    if horizontal:
      plt.subplots(1,2, figsize=(14,5))
    else:
      plt.subplots(2,1)
  else:
    plt.subplots(1)
    
  plt.suptitle(plot_title)

  if compare_with_sound is not None:
    if horizontal:
      plt.subplot(1,2,1)
    else:
      plt.subplot(2,1,1)
    plt.title("Original")

  
  f1, periodogram1 = signal.periodogram(sound, sr)
  plt.semilogy(f1, periodogram1)
  plt.ylim([1e-7, 1e2])
  plt.ylabel('Magnitude (norm)')
  if compare_with_sound is None or horizontal:
    plt.xlabel('Frequency [Hz]')

  if compare_with_sound is not None:
    if horizontal:
      plt.subplot(1,2,2)
    else:
      plt.subplot(2,1,2)

    subplot_title = "Preprocessed"
    if preprocessing_name is not None:
      subplot_title += " ("+preprocessing_name+")"
    plt.title(subplot_title)

    f2, periodogram2 = signal.periodogram(compare_with_sound, sr)
    plt.semilogy(f2, periodogram2)

    plt.ylim([1e-7, 1e2])
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude (norm)')
    plt.subplots_adjust(hspace= 0.4)
  
  if show: 
    plt.show()
  
  if save_to_dir is not None:
    path = os.path.join(save_to_dir,plot_title.replace("\n", " ")+".png")
    plt.savefig(path)
      
def show_audio_preprocessing(original_clip, preprocessors, save_clips_to_dir, save_plots_to_dir, horizontal = False,
                              sound_file_name = None, sound_class = None, sample_rate = 22050,
                              show = False,
                              spectrograms = True, waveplots = True, periodograms = True, 
                              comparison = True):

  output_dir = os.path.join(save_plots_to_dir, "No augmentation - Plots")
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  if not os.path.exists(save_clips_to_dir):
    os.makedirs(save_clips_to_dir)
  
  plot_title = ""
  if sound_file_name is not None:
    plot_title += str(sound_file_name)
  if sound_class is not None:
    plot_title += " ("+str(sound_class)+")"

  plot_periodogram(original_clip, show = show, save_to_dir = output_dir, plot_title = plot_title)
  plot_sound_waves(original_clip, show = show, save_to_dir = output_dir, plot_title = plot_title)
  plot_sound_spectrogram(original_clip, show = show, save_to_dir = output_dir, plot_title = plot_title)

  plot_title += "\n"
  for preprocessor in preprocessors:
    output_dir = os.path.join(save_plots_to_dir, preprocessor.name+" - Plots")
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

    
    preprocessing_value = preprocessor.values[0]
    preprocessed_clip = preprocessor(original_clip, value = preprocessing_value)
    scipy.io.wavfile.write(os.path.join(save_clips_to_dir,sound_class,preprocessor.name+" - "+sound_file_name), sample_rate, preprocessed_clip)

    preprocessing_name = preprocessor.name + ": " + str(preprocessing_value)
    plot_periodogram(original_clip, show = show, compare_with_sound = preprocessed_clip, preprocessing_name = preprocessing_name, horizontal = horizontal, save_to_dir = output_dir, plot_title = plot_title)
    plot_sound_waves(original_clip, show = show, compare_with_sound = preprocessed_clip, preprocessing_name = preprocessing_name, horizontal = horizontal, save_to_dir = output_dir, plot_title = plot_title)
    plot_sound_spectrogram(original_clip, show = show, compare_with_sound = preprocessed_clip, preprocessing_name = preprocessing_name, horizontal = horizontal, save_to_dir = output_dir, plot_title = plot_title)


    