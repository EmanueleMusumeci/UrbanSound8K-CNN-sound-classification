import os, threading

import dill

import torch
from torch import optim, nn

from Dataset import SoundDatasetFold
from DataLoader import DataLoader

from data_augmentation.image_transformations import *
from data_augmentation.image_transformations import *

from nn.paper_convolutional_model import PaperConvolutionalNetwork

from utils.evaluation_utils import *

"""
Args:
    - instance_name: name of the training instance (also name of the checkpoint directory)
    - batch_size
    - train_loader: iterable providing training set data in batches 
    - test_loader: iterable providing test set data in batches 
    - model: model to train
    - loss_function
    - optimizer: optimizer used for loss_function optimization
    - device: device to train on
    - model_dir: directory that will contain all checkpoints
OPTIONAL:
    - lr_scheduler: learning rate scheduler
    - cnn (default: True): determines if we're wrapping the sound classification cnn or the ffn
"""
class Trainer:
    def __init__(
        self,
        instance_name,
        batch_size,
        train_loader,
        test_loader,
        model,
        loss_function,
        optimizer,
        device,
        model_dir,
        lr_scheduler=None,
        cnn = True
    ):
        self.instance_name = instance_name
        self.model = model


        self.optimizer = optimizer
        self.last_epoch = 0
        self.device = device
        self.model.to(self.device)
        self.checkpoint_path = model_dir

        self.train_loader = train_loader
        self.test_loader = test_loader
        
        self.batch_size = batch_size

        self.loss_function = loss_function

        self.lr_scheduler = lr_scheduler   

        self.cnn = cnn
    
    """
    Contains the training loop
    Args:
    - epochs: number of training epochs.
    OPTIONAL:
    - save_test_scores_every: number of epochs intercurring between two evaluations on test set
    - save_test_scores_every: number of epochs intercurring between two evaluations on training set
    - save_model_every: number of epochs intercurring between two model checkpoints
    - compute_gradient_statistics: add gradient statistics to the saved scores (used to plot gradient flows later)
    """
    def train(self, epochs, save_test_scores_every=0, save_train_scores_every=0, save_model_every=0, compute_gradient_statistics = False):

                
        self.save_model_structure()

        print("Beginning training process: ")

        for epoch in range(self.last_epoch,self.last_epoch+epochs):       
            self.model.train()

            first_batch = True
            total_batches = 0     
            total_samples = 0
            running_loss = 0
            batch_losses = []
                        
            for batch in self.train_loader:
                if batch is None:
                    break

                self.optimizer.zero_grad()

                if self.cnn:
                    preprocessed_spectrogram = batch["preprocessed_spectrogram"].to(self.device)
                    predictions = self.model(preprocessed_spectrogram)
                else:
                    mfccs = batch["mfccs"].to(self.device)
                    chroma = batch["chroma"].to(self.device)
                    mel = batch["mel"].to(self.device)
                    contrast = batch["contrast"].to(self.device)
                    tonnetz = batch["tonnetz"].to(self.device)
                    predictions = self.model(mfccs, chroma, mel, contrast, tonnetz)
                    
                labels = batch["class_id"].to(self.device)             
                
                loss = self.loss_function(predictions, labels)

                loss.backward()
                
                #Only for first batch, plot the gradient flow
                if first_batch and compute_gradient_statistics: 
                    gradient_stats = self.get_gradient_stats(self.model.named_parameters())
                    first_batch = False

                self.optimizer.step()

                #Save total batch losses
                batch_loss = loss.item() * len(batch)
                batch_losses.append(batch_loss) 
                running_loss += batch_loss
                total_samples += len(batch)
                total_batches += 1


            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            #Batch scores computation
            #Average loss per sample
            average_sample_loss = running_loss/total_samples
            #Average loss per batch (loss.item() returns the average sample loss for a batch)
            average_batch_loss = np.sum(batch_losses)/total_batches

            #Print epoch stats
            print('Epoch: {} Classification Loss = {:0.4f}'.format(epoch, average_sample_loss))
            
            #Compute and save epoch scores on test_set
            if save_train_scores_every!=0 and epoch%save_train_scores_every==0:
                #Evaluate on validation set     
                results = self.evaluate(log_output_directory=os.path.join(self.checkpoint_path, self.instance_name), train = True)

                #Update loss info of computed scores
                results["batch_loss"] = average_batch_loss
                results["loss"] = average_sample_loss
                if compute_gradient_statistics:
                    results["gradient_stats"] = gradient_stats

                #Save computed scores for later inspection/model-selection
                current_epoch_scores = {"audio_classification" : results,
                                        "Epoch": epoch
                                        }

                self.save_scores(current_epoch_scores, train=True)

            #Compute and save epoch scores on test_set
            if save_test_scores_every!=0 and epoch%save_test_scores_every==0:
                #Evaluate on validation set     
                results = self.evaluate(log_output_directory=os.path.join(self.checkpoint_path, self.instance_name))

                #Update loss info of computed scores
                results["batch_loss"] = average_batch_loss
                results["loss"] = average_sample_loss
                if compute_gradient_statistics:
                    results["gradient_stats"] = gradient_stats

                #Save computed scores for later inspection/model-selection
                current_epoch_scores = {"audio_classification" : results,
                                        "Epoch": epoch
                                        }

                self.save_scores(current_epoch_scores)

            #Save model weights (every save_model_every epochs or if this is the last epoch)
            if (save_model_every!=0 and epoch%save_model_every==0) or epoch==self.last_epoch+epochs-1:
                self.save_model_weights()

            self.last_epoch += 1    
            #return self.scores

    '''
    Evaluate model performance on various performance metrics
    Args:
    OPTIONAL:
    - compute_confusion_matrix
    - train: test on training set (True)
    - log_output_directory: allows printing a score log if not None
    - print_scores: print scores on console (True)
    '''

    def evaluate(self, compute_confusion_matrix=False, log_output_directory=None, print_scores=True, train = False):

        if train:
          loader = self.train_loader
        else:
          loader = self.test_loader

        self.model.eval()

        def count(l):
            d = {}
            for e in l:
                d[e] = 1 + d.get(e, 0)
            return d
        
        all_class_predictions = []
        all_class_labels = []

        with torch.no_grad():

            for batch in loader:

                labels = batch["class_id"].cpu().detach().numpy()
                
                if self.cnn:
                    preprocessed_spectrogram = batch["preprocessed_spectrogram"].to(self.device)
                    predictions = self.model(preprocessed_spectrogram)
                else:
                    mfccs = batch["mfccs"].to(self.device)
                    chroma = batch["chroma"].to(self.device)
                    mel = batch["mel"].to(self.device)
                    contrast = batch["contrast"].to(self.device)
                    tonnetz = batch["tonnetz"].to(self.device)
                    predictions = self.model(mfccs, chroma, mel, contrast, tonnetz)
                    

                predictions = torch.argmax(predictions, axis= 1).detach().cpu().numpy()

                decoded_labels = loader.dataset.decode_class_names(labels)
                decoded_predictions = loader.dataset.decode_class_names(predictions)

                #Accumulate batch entries
                all_class_predictions.extend(decoded_predictions)
                all_class_labels.extend(decoded_labels)
            
        audio_classification_results = evaluate_class_prediction(all_class_labels, all_class_predictions)
        audio_classification_table = print_table('Audio classification', audio_classification_results, fields=["accuracy", "precision", "recall", "f1"])
        audio_classification_distribution = print_distribution(audio_classification_results)
        if print_scores: print(audio_classification_table)
        if print_scores: print(audio_classification_distribution)

        if log_output_directory is not None:
            ResultsWriterThread(all_class_predictions, all_class_labels, "audio_classification", 
                                audio_classification_table,log_output_directory, 
                                self.instance_name+"_audio_classifications", epoch=self.last_epoch, overwrite=False).start()

        return audio_classification_results

    '''
    Serializes all data necessary to load this model (includes all objects that concur to training)
    '''
    def save_model_structure(self):
        path = os.path.join(self.checkpoint_path,self.instance_name)
        
        try:
            os.makedirs(path)
        except OSError:
            print("Directory: ",path," already exists.")
        
        #How checkpoints were previously saved (for backwards compatibility I left this here)
        torch.save({
            'instance_name': self.instance_name,
            'device': self.device,

            'model': self.model,
            'optimizer': self.optimizer,
            'lr_scheduler': self.lr_scheduler
        }, os.path.join(path,"model_structure.pt"), pickle_module=dill)

    '''
    Saves only weights
    '''
    def save_model_weights(self):
        path = os.path.join(self.checkpoint_path,self.instance_name)
        try:
            os.makedirs(path)
        except OSError:
            print("Directory: ",path," already exists.")
        
        if self.lr_scheduler is None:
            torch.save({
                'epoch': self.last_epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, os.path.join(path,"_{}.pt".format(self.last_epoch)), pickle_module=dill)
        else:
            torch.save({
                'epoch': self.last_epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            }, os.path.join(path,"_{}.pt".format(self.last_epoch)), pickle_module=dill)

    '''
    Save model scores
    '''
    def save_scores(self, scores, train=False):
        epoch = scores["Epoch"]
        
        if train:
          directory = os.path.join(self.checkpoint_path,self.instance_name,"scores_on_train")
        else:
          directory = os.path.join(self.checkpoint_path,self.instance_name,"scores_on_test")

        complete_file_path = os.path.join(directory,"_"+str(epoch)+".scores")
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        with open(complete_file_path, mode="wb+") as f:
            dill.dump(scores,f)
        
        #write a small log file for scores
        complete_file_path = complete_file_path+".log"
        with open(complete_file_path, mode="w") as f:
            for task_name, task_score in scores.items():
                if task_score is None: continue
                if isinstance(task_score,int) or isinstance(task_score,float): 
                    f.write(task_name+": "+str(task_score))
                    continue
                
                f.write(print_table(task_name, task_score, fields=task_score.keys()))
                try:
                    f.write(print_distribution(task_score))
                    f.write("Confusion matrix:\n"+print_confusion_matrix(task_score))
                except:
                    pass
                f.write("\n")

    '''
    Returns an instance of Trainer by loading a previously saved model structure and checkpoint
    Args:
    - train_loader: dataloader used in this trainer
    - test_loader: dataloader used in this trainer
    - instance_name: instance name of the training session (also the name of the subdirectory containing the model)
    - model_dir: directory containing all models
    - checkpoint_epoch: training epoch to load
    OPTIONAL
    - device
    - is_cnn (default: True): are we in cnn training mode
    Returns
    - an instance of Trainer containing the loaded training session
    ''' 
    @classmethod
    def load(cls, train_loader, test_loader, instance_name, model_dir, checkpoint_epoch, 
                device="cuda", batch_size = 128, loss_function=None, is_cnn = True):

        if loss_function is None:
            loss_function = nn.CrossEntropyLoss()

        #find model file that refers to this epoch
        model_path = os.path.join(model_dir,instance_name)

        model_structure=None
        trainer = None

        for filename in os.listdir(model_path):
            if not os.path.isfile(os.path.join(model_path,filename)):
                continue
            elif filename.endswith("structure.pt"):
                model_structure = torch.load(os.path.join(model_path,filename), map_location=torch.device(device), pickle_module=dill)
                instance_name = model_structure["instance_name"]        

                print("Attempting to load model: "+instance_name+"\n")
                
                model = model_structure["model"]
                print("Model structure loaded:\n"+str(model))

                
                model.to(device=device)

                print("Loading optimizer\n")
                optimizer = model_structure["optimizer"]
                print(optimizer)
                
                try:
                    lr_scheduler = model_structure["lr_scheduler"] 
                    print(lr_scheduler)
                except Exception as er:
                    lr_scheduler = None

                trainer = Trainer(
                                    instance_name,
                                    batch_size,
                                    train_loader,
                                    test_loader, 
                                    model, 
                                    loss_function,
                                    optimizer, 
                                    device, 
                                    model_dir, 
                                    lr_scheduler = lr_scheduler,
                                    cnn = is_cnn
                                 )
                            
                print("Trainer state restored\n")
                break

        if model_structure==None:
            raise FileNotFoundError

        print("Loading model checkpoint from epoch "+str(checkpoint_epoch)+" at path: "+model_path+"\n")

        checkpoint = None
        for filename in os.listdir(model_path):
            if not os.path.isfile(os.path.join(model_path,filename)):
                continue
            elif filename.startswith("_") and filename.endswith(str(checkpoint_epoch)+".pt"):
                print("Checkpoint found:" +filename)
                checkpoint = torch.load(os.path.join(model_path,filename), map_location=torch.device(device), pickle_module=dill)
                if checkpoint["epoch"]==checkpoint_epoch:
                    trainer.model.load_state_dict(checkpoint["model_state_dict"])
                    trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    if trainer.lr_scheduler is not None:
                        trainer.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
                    trainer.last_epoch = checkpoint_epoch+1
                    try:
                        self.scores = checkpoint["scores"]
                    except:
                        pass
                    break            

        if checkpoint is None:
            raise FileNotFoundError

        print("Loading process completed\n")
        return trainer

    '''
    Loads only the model weights in the current training session (looking for \
    the subdirectory with name 'self.instance_name' of the current instance in 'path') 
    Args:
    - checkpoint_epoch: training epoch to load
    OPTIONAL
    - path: path of the directory containing all model subdirectories
    - device
    '''
    def load_without_structure(self, checkpoint_epoch, path=None, device="cuda"):
        if path is None:
            path = os.path.join(self.checkpoint_path,self.instance_name)
        
        checkpoint = None
        for filename in os.listdir(path):
            #print(filename)
            if not os.path.isfile(os.path.join(path,filename)):
                continue
            elif filename.endswith("_"+str(checkpoint_epoch)+".pt"):
                checkpoint = torch.load(os.path.join(path,filename), map_location=torch.device(device), pickle_module=dill)
                if checkpoint["epoch"]==checkpoint_epoch:
                    assert self.mode == model.mode, "Required mode is :"+str(self.mode)+" while model mode is: "+str(model.mode)
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                    if trainer.lr_scheduler is not None:
                        trainer.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
                    self.last_epoch = checkpoint_epoch+1
                    try:
                        self.dev_scores = checkpoint["scores"]
                    except:
                        pass
                    break            

        if checkpoint is None:
            raise FileNotFoundError
    
    '''
    Computes various statistics about the gradients flowing through the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow
    '''
    def get_gradient_stats(self, named_parameters):
        gradient_stats = {}
        layer_names = []
        for layer_name, layer_parameters in named_parameters:
            if(layer_parameters.requires_grad) and ("bias" not in layer_name):
                gradient_stats[layer_name] = {}
                layer_names.append(layer_name)
                gradient_stats[layer_name]["avg_grads"] = layer_parameters.grad.abs().mean().cpu().detach().numpy()
                gradient_stats[layer_name]["max_grads"] = layer_parameters.grad.abs().max().cpu().detach().numpy()
                gradient_stats[layer_name]["min_grads"] = layer_parameters.grad.abs().min().cpu().detach().numpy()
        return gradient_stats

'''
Runnable that creates a small "preview" of the scores for a certain epoch
'''
class ResultsWriterThread(threading.Thread):
    def __init__(self, predictions, gold, key, results, directory, filename, 
                epoch=None, overwrite=True, verbose=False):
        threading.Thread.__init__(self)
        self.predictions = predictions
        self.gold = gold
        self.key = key
        self.results = results
        self.directory = directory
        self.filename = filename
        self.overwrite = overwrite
        self.verbose = verbose
        if epoch is not None and not overwrite:
            self.filename = self.filename+".Epoch_"+str(epoch)
            self.instance_name = self.filename
    def run(self):
        complete_file_path = os.path.join(self.directory,self.instance_name)+".log"
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        
        with open(complete_file_path, mode="w+", encoding="utf-8") as f:
            if self.verbose: print("Results writer started writing in file: "+complete_file_path)
            f.write(self.results)
            for id, (prediction, label) in enumerate(zip(self.predictions, self.gold)):
                f.write("Image: "+str(id)+"\tClass prediction:"+str(prediction)+"\tClass label:"+str(label)+"\n")
        return