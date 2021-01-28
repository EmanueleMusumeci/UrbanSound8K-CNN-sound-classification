
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

import PyTorchVisualizations
#from PyTorchVisualizations.src.gradcam import GradCam
#from PyTorchVisualizations.src.misc_functions import save_class_activation_images
from PyTorchVisualizations.src.misc_functions import save_class_activation_images

import librosa

import scipy

from scipy import signal



'''
Displays a wave plot for the input raw sound (using the Librosa library)
'''
def plot_sound_waves(sound, sound_file_name = None, sound_class=None, show=False, sr=22050):
    plot_title = "Wave plot"
    
    if sound_file_name is not None:
        plot_title += "File: "+sound_file_name
    
    if sound_class is not None:
        plot_title+=" (Class: "+sound_class+")"
    
    plot = plt.figure(plot_title)
    librosa.display.waveplot(np.array(sound),sr=sr)
    plt.title(plot_title)
    
    if show:
        plt.show()

def plot_sound_spectrogram(sound, sound_file_name = None, sound_class=None, show = False, log_scale = False, hop_length=512, sr=22050, colorbar_format = "%+2.f dB", title=None):
    if title is None:
        plot_title = title
    else:
        plot_title = "Spectrogram"
        
        if sound_file_name is not None:
            plot_title += "File: "+sound_file_name
        
        if sound_class is not None:
            plot_title+=" (Class: "+sound_class+")"
    
    sound = librosa.stft(sound, hop_length = hop_length)
    sound = librosa.amplitude_to_db(np.abs(sound), ref=np.max)

    if log_scale:
        y_axis = "log"
    else:
        y_axis = "linear"

    plot = plt.figure(plot_title)
    librosa.display.specshow(sound, hop_length = hop_length, x_axis="time", y_axis=y_axis)

    plt.title(plot_title)
    plt.colorbar(format=colorbar_format)
    
    if show:
        plt.show()
    
    return plot

#from https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.periodogram.html
def plot_periodogram(sound, sound_file_name = None, sound_class=None, show = False, sr=22050, title=None):
    f, Pxx_den = signal.periodogram(sound, sr)
    plot = plt.figure()
    plt.semilogy(f, Pxx_den)
    plt.ylim([1e-7, 1e2])
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude (norm)')

    if show:
        plt.show()
    return plot


###########
#  PLOTS  #
###########

'''
Utils to generate all plots and graphical renders needed for the presentation
'''
def plot_scores(model_name, model_dir, tasks={"audio_classification" : "Audio classification"},
                metrics={"F1-macro":["f1"]}, plot_confusion_matrix=True, 
                from_epoch=0, to_epoch=0, epochs_skip=0, save_to_file=False,
                xticks_step=0, combine_tasks=False, increase_epoch_labels_by_one=False, 
                title_prefix=None, color = None):
  '''
  Plots the requested performance metrics from the score history of the current model
  (used to generate the graphs in the report)
  Args:
    - model_name: name of the model that generated the scores to be plotted
    - model_dir: directory containing the models
    OPTIONAL
    - tasks: dictionary containing tasks whose score is going to be plotted and
             the name to print on the plot
    - plot_confusion_matrix: (default: True)
    - from_epoch: plot scores starting from a certain episode (default: 0)
    - to_epoch: plot scores starting until a certain episode (default: 0, plot all scores)
    - epochs_skip: allows skipping n epochs every plotted point (if scores where saved
                   every n epochs) (default: 0)
    - save_to_file: save plots to file (default: False)
    - xticks_step: used to print a "tick" on the graph x axis every n ticks 
      (with a value of 5 we would have 0,5,10... on the x axis)
    - combine_tasks: allows plotting scores for all tasks on the same plot (False)
  '''

  plt.close("all")

  #1) Load scores
  scores_directory = os.path.join(model_dir,model_name)
  scores_directory = os.path.join(model_dir,model_name)
  if os.path.exists(os.path.join(scores_directory,"scores_on_train")):
    scores_directory = os.path.join(scores_directory,"scores_on_train")
  else:
    scores_directory = os.path.join(scores_directory,"scores_on_test")


  losses = {}
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
      
  plot_dir = os.path.join(model_dir,"plots")
  if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
  for k,plot in plots.items():
    #plot.show()
    if save_to_file:
      path = os.path.join(plot_dir,model_name+" - "+k)+".png"
      plot.savefig(path)
  

  if plot_confusion_matrix:
    for task_key, task_header in tasks.items():
      assert task_key in scores.keys(), "Scores for task "+task_key+" not found"
      try:
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
        fig, axes = plt.subplots(figsize=(10,10))  

        #seaborn.set(font_scale=2)
        ax = seaborn.heatmap(confusion_matrix, 
                        norm = LogNorm(vmin=min_val, vmax=max_val),
                        cbar_kws={"shrink": 0.5, "ticks":[0,1,10,1e2,1e3,1e4,1e5]}, 
                        annot=True, ax = axes, fmt='g'
                        ) #annot=True to annotate cells, fmt='g' to avoid scientific notation
        ax.figure.subplots_adjust(left=0.3, bottom=0.3)

        plt.xticks(rotation=90) 
        plt.yticks(rotation=0) 


        # labels, title and ticks
        axes.set_xlabel('Predicted labels', size = 10)
        axes.set_ylabel('True labels', size = 10) 
        
        if title_prefix is not None:
            conf_matrix_title = title_prefix
        else:
            conf_matrix_title = model_name
        axes.set_title(conf_matrix_title+"\nEpoch: "+str(best_epoch)+'\nConfusion Matrix') 

        axes.xaxis.set_ticklabels(labels, size = 10)
        axes.yaxis.set_ticklabels(labels, size = 10)

        plots[task_header+"_Confusion_matrix"] = fig
        

        #fig.show()
        if save_to_file:
          path = os.path.join(plot_dir,model_name+"_Confusion_matrix")+".png"
          fig.savefig(path)
        
      except Exception as e:
        print(e)
        pass

  plt.clf()

def plot_scores_from_multiple_dirs(
                model_name, model_dir, score_dirs, tasks=None,
                metrics={"F1-macro":["f1"]}, plot_confusion_matrix=True, 
                from_epoch=0, to_epoch=0, epochs_skip=0, save_to_file=False,
                xticks_step=0, combine_tasks=False, increase_epoch_labels_by_one=False, 
                title_prefix=None, colors = None, switch_labels = False):
  '''
  Plots the requested performance metrics from the score history of the current model
  (used to generate the graphs in the report)
  Args:
    - model_name: name of the model that generated the scores to be plotted
    - model_dir: directory containing the models
    OPTIONAL
    - tasks: dictionary containing tasks whose score is going to be plotted and
             the name to print on the plot
    - plot_confusion_matrix: (default: True)
    - from_epoch: plot scores starting from a certain episode (default: 0)
    - to_epoch: plot scores starting until a certain episode (default: 0, plot all scores)
    - epochs_skip: allows skipping n epochs every plotted point (if scores where saved
                   every n epochs) (default: 0)
    - save_to_file: save plots to file (default: False)
    - xticks_step: used to print a "tick" on the graph x axis every n ticks 
      (with a value of 5 we would have 0,5,10... on the x axis)
    - combine_tasks: allows plotting scores for all tasks on the same plot (False)
  '''

  plt.close("all")

  #1) Load scores

  losses = {}
  all_scores = {}
  epochs_list = []
  
  best_epoch = 0 #Find best epoch based on f1 score
  best_f1 = 0
  
  for scores_header, scores_subdir in score_dirs.items():
    scores = {}
    scores_directory = os.path.join(model_dir, model_name, scores_subdir)
    for filename in natsorted(os.listdir(scores_directory)):
      #read files in alphabetical order
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
                    if scores_entry[key]["f1"] > best_f1:
                      best_epoch = scores_entry["Epoch"]
                      best_f1 = scores_entry[key]["f1"]
                      scores[key]["confusion matrix"] = score_value
                elif score_name=="distribution":
                    if scores_entry[key]["f1"] > best_f1:
                      best_epoch = scores_entry["Epoch"]
                      best_f1 = scores_entry[key]["f1"]
                      scores[key]["confusion matrix"] = score_value
                else:
                  if score_name not in scores[key].keys():
                    scores[key][score_name] = {}
                  scores[key][score_name][scores_entry["Epoch"]] = scores_entry[key][score_name]
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
              
  plot_dir = os.path.join(model_dir,"plots")
  if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
  for k,plot in plots.items():
    #plot.show()
    if save_to_file:
      path = os.path.join(plot_dir,model_name+" - Train_Test comparison - "+k)+".png"
      plot.savefig(path)
  
  plt.clf()
    

def comparative_plots(model_names, model_dir, 
                      tasks={"audio_classification" : "Audio classification"},
                      metrics={"F1-macro":["f1"]}, 
                      from_epoch=0, to_epoch=0, epochs_skip=0, save_to_file=False,
                      xticks_step=0, increase_epoch_labels_by_one=False,
                      title_prefix=None
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
    - save_to_file: save plots to file (default: False)
    - xticks_step: used to print a "tick" on the graph x axis every n ticks 
      (with a value of 5 we would have 0,5,10... on the x axis)
  '''

  plt.close("all")

  #1) Load scores for each model
  scores = {}

  #Epochs list is going to hold the available epochs in the interval [from_epoch, to_epoch]
  #for the model with most epochs (the one that has been trained longest)
  epochs_list = []

  for model_name, model_header in model_names.items():
    scores_directory = os.path.join(model_dir,model_name)
    if os.path.exists(os.path.join(scores_directory,"scores_on_train")):
      scores_directory = os.path.join(scores_directory,"scores_on_train")
    else:
      scores_directory = os.path.join(scores_directory,"scores_on_test")

    scores[model_name] = {}

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

          if scores_entry["Epoch"] not in epochs_list:
            epochs_list.append(scores_entry["Epoch"])
          for key, value in scores_entry.items():
            if key=="Epoch": continue
            else:
              if key not in scores[model_name].keys():
                scores[model_name][key] = {}
              if scores_entry[key] is None: continue
              for score_name, score_value in scores_entry[key].items():
                if score_name=="confusion matrix": continue
                elif score_name=="distribution":
                  continue
                else:
                  if score_name not in scores[model_name][key].keys():
                    scores[model_name][key][score_name] = {}
                  scores[model_name][key][score_name][scores_entry["Epoch"]] = scores_entry[key][score_name]

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
      for model_name, model_scores in scores.items():

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
  
  plot_dir = os.path.join(model_dir,"comparative_plots")
  
  if save_to_file:
    if not os.path.exists(plot_dir):
      os.makedirs(plot_dir)
  for k,plot in plots.items():
    #plot.show()
    if save_to_file:
      path = os.path.join(plot_dir,title_prefix+" - "+k)+".png"
      plot.savefig(path)

  plt.close(plot)

def print_best_epoch_scores(model_name, model_dir, metrics, print_scores=True):
    
    #1) Load scores
    scores_directory = os.path.join(model_dir,model_name)
    scores_directory = os.path.join(model_dir,model_name)
    if os.path.exists(os.path.join(scores_directory,"scores_on_train")):
      scores_directory = os.path.join(scores_directory,"scores_on_train")
    else:
      scores_directory = os.path.join(scores_directory,"scores_on_test")


    losses = {}
    scores = {}
    epochs_list = []
        
    best_f1 = 0
    best_epoch = 0

  #read files in alphabetical order
    for filename in natsorted(os.listdir(scores_directory)):
        if not os.path.isfile(os.path.join(scores_directory,filename)):
          continue
        elif filename.endswith(".scores"):
          scores_path = os.path.join(scores_directory, filename)
          with open(scores_path, "rb") as f:
            scores_entry = dill.load(f)
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

    best_scores = {}
    best_scores_str = "-"*40 + "\n" + model_name + "\n" + "-"*40 + "\n"
    best_scores_str += "Epoch: "+str(best_epoch)+"\n"

    for score_name, values in scores["image_classification"].items():
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


def create_gradient_flow_gif(model_name, model_dir, cropX = None, cropY = None):
    filenames = []
    for filename in natsorted(os.listdir(os.path.join(model_dir,model_name))):
        if filename.startswith("Gradient flow"):
            filenames.append(os.path.join(model_dir,model_name, filename))

    images = []
    for filename in filenames:
        image = imageio.imread(filename)
        if cropX is not None:
          image = image[cropX[0]:cropX[1],:,:]
        if cropY is not None:
          image = image[:,cropY[0]:cropY[1],:]
        images.append(image)
    imageio.mimsave(os.path.join(model_dir, "plots",model_name+"_Gradient_flow.gif"), images)

def show_preprocessing(transformations, image, title_prefix="", progressive=True, save_to_dir=None):
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

def plot_class_distributions(distributions, save_path=None):
    fig = plt.figure(figsize=(18,7))
    seaborn.barplot(data = pd.DataFrame(distributions, index=[0]).melt(), 
                    x = "variable", y="value", hue="variable").set_title('Class distribution')

    if save_path is not None:
        fig.savefig(os.path.join(save_path,"class_distribution_plot.png"))
    else:
        fig.show()
        
def visualize_features(model, image, image_label_idx, training_preprocessing_pipeline,
                      validation_preprocessing_pipeline, save_to_dir, filename=None, layer=0):
  if not os.path.exists(save_to_dir):
    os.makedirs(save_to_dir)

  # Get params  
  training_preprocessed_image = training_preprocessing_pipeline(image).unsqueeze(0).cuda()
  validation_preprocessed_image = validation_preprocessing_pipeline(image)
  # Grad cam
  grad_cam = GradCam(model.cuda(), target_layer=layer)
  # Generate cam mask
  cam = grad_cam.generate_cam(training_preprocessed_image, image_label_idx)

  if filename is None:
    filename = type(model).__name__
  # Save mask
  save_class_activation_images(validation_preprocessed_image, cam, filename, directory=save_to_dir)
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
  imageio.mimsave(os.path.join(save_to_dir,gif_name)+".gif", images)

def visualize_features_on_layers(model, image, image_label_idx, 
                                training_preprocessing_pipeline, validation_preprocessing_pipeline,
                                save_to_dir,
                                from_layer=0, to_layer=0,
                                filename_prefix=None, make_gif=True):
  assert from_layer<=to_layer, "Impossible range: ("+str(from_layer)+", "+str(to_layer)+")"
  
  if not os.path.exists(save_to_dir):
    os.makedirs(save_to_dir)

  if to_layer==0:
    to_layer = len(model.feature_extractor._modules)
  for layer in range(from_layer, to_layer):
    try:
      visualize_features(model, image, image_label_idx, training_preprocessing_pipeline,
                        validation_preprocessing_pipeline, save_to_dir, filename=filename_prefix+str(layer), layer=layer)
    except Exception as e:
      print(e)
      pass
  
  if make_gif:
    create_gif(save_to_dir, "", file_endswith="On_Image.png", gif_name="Layer_activations", save_to_dir = save_to_dir)

