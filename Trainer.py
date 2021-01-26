import os, threading

import dill

import torch
from torch import optim, nn

try:
    from Dataset import SoundDatasetFold
    from DataLoader import DataLoader

    from data_augmentation.image_transformations import *
    from data_augmentation.image_transformations import *

    from nn.feed_forward_model import FeedForwardNetwork
    from nn.convolutional_model import ConvolutionalNetwork

    from utils.evaluation_utils import *
except:
    pass

class Trainer:
    def __init__(
        self,
        instance_name,
        batch_size,
        train_loader,
        test_loader,
        index_to_class,
        model,
        loss_function,
        optimizer,
        device,
        model_dir,
        lr_scheduler=None,
        cnn = True
    ):
        """
        Args:
            - train_loader: iterable providing training set data in batches 
            - dev_loader: iterable providing dev set data in batches 
            - test_loader: iterable providing test set data in batches 
            - model: model to train
            - optimizer: optimizer used for loss_function optimization
            - device: device to train on
        """
        self.instance_name = instance_name
        self.model = model


        self.optimizer = optimizer
        self.last_epoch = 0
        self.device = device
        self.model.to(self.device)
        self.checkpoint_path = model_dir

        self.train_loader = train_loader
        self.test_loader = test_loader

        self.index_to_class = index_to_class
        
        self.batch_size = batch_size

        self.loss_function = loss_function

        self.lr_scheduler = lr_scheduler   

        self.cnn = cnn

    def train(self, epochs, save_test_scores_every=0, save_train_scores_every=0, save_model_every=0, compute_gradient_statistics = False):
        """
        Contains the training loop
        Args:
        - epochs: number of training epochs.
        OPTIONAL:
        - save_test_scores_every: number of epochs intercurring between two evaluations on test set
        - save_model_every: number of epochs intercurring between two model checkpoints
        """
                
        self.save_model_structure()

        print("Beginning training process: ")

        for epoch in range(self.last_epoch,self.last_epoch+epochs):       
            self.model.train()

            first_batch = True
            #if total_batches>0:
            #    progress_bar = tqdm(total=len(total_batches), desc='Epoch: '+str(epoch), position=0)
            #else:
            #    progress_bar = None
            total_batches = 0     
            total_samples = 0
            running_loss = 0
            batch_losses = []
            
            #self.train_loader.dataset.test_mode = False
            
            for batch in self.train_loader:
                if batch is None:
                    break

                self.optimizer.zero_grad()

                if self.cnn:
                    #original_spectrogram = batch["original_spectrogram"]
                    preprocessed_spectrogram = batch["preprocessed_spectrogram"].to(self.device)
                    predictions = self.model(preprocessed_spectrogram)
                else:
                    mfccs = batch["mfccs"].to(self.device)
                    chroma = batch["chroma"].to(self.device)
                    mel = batch["mel"].to(self.device)
                    contrast = batch["contrast"].to(self.device)
                    tonnetz = batch["tonnetz"].to(self.device)
                    predictions = self.model(mfccs, chroma, mel, contrast, tonnetz)
                    
                #labels = batch["class_id"].to(self.device)                
                labels = batch["class_id"].to(self.device)             
                
                loss = self.loss_function(predictions, labels)

                #loss.backward(retain_graph=True)
                loss.backward()
                
                #Only for first batch, plot the gradient flow
                if first_batch and compute_gradient_statistics: 
                    #self.plot_gradient_flow(self.model.named_parameters(), save_to_dir = os.path.join(self.checkpoint_path,self.instance_name))
                    gradient_stats = self.get_gradient_stats(self.model.named_parameters())
                    first_batch = False

                self.optimizer.step()

                #Save total batch losses
                batch_loss = loss.item() * len(batch)
                batch_losses.append(batch_loss) 
                running_loss += batch_loss
                total_samples += len(batch)
                total_batches += 1

                #if progress_bar is not None:
                #    progress_bar.update(1)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            #if progress_bar is not None:
            #    progress_bar.close()

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

    def evaluate(self, compute_confusion_matrix=False, log_output_directory=None, print_scores=True, train = False):
        '''
        Evaluate model performance on precision, recall and f1-score
        Args:
        OPTIONAL:
        - compute_confusion_matrix
        - dev: test on dev set (True)
        - log_output_directory: allows printing a score log if not None
        - print_scores: print scores on console (True)
        '''

        if train:
          #self.train_loader.dataset.test_mode = True
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

            #progress_bar = tqdm(total=len(loader), desc='Evaluating', position=0)
            for batch in loader:

                labels = batch["class_id"].cpu().detach().numpy()
                
                if self.cnn:
                    #original_spectrogram = batch["original_spectrogram"]
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

                decoded_labels = self.decode_class_names(labels)
                decoded_predictions = self.decode_class_names(predictions)

                #Accumulate batch entries
                all_class_predictions.extend(decoded_predictions)
                all_class_labels.extend(decoded_labels)

                #progress_bar.update(1)

            #progress_bar.close()
            
        audio_classification_results = evaluate_class_prediction(all_class_labels, all_class_predictions)
        audio_classification_table = print_table('Image classification', audio_classification_results, fields=["accuracy", "precision", "recall", "f1"])
        audio_classification_distribution = print_distribution(audio_classification_results)
        if print_scores: print(audio_classification_table)
        #if print_scores: print_confusion_matrix(audio_classification_results)
        if print_scores: print(audio_classification_distribution)

        if log_output_directory is not None:
            ResultsWriterThread(all_class_predictions, all_class_labels, "audio_classification", 
                                audio_classification_table,log_output_directory, 
                                self.instance_name+"_audio_classifications", epoch=self.last_epoch, overwrite=False).start()

        return audio_classification_results

    def decode_class_names(self, class_indices):
        return [self.index_to_class[idx] for idx in class_indices]

    def save_model_structure(self):
        '''
        Serializes all data necessary to restore this model (includes all objects that concur to training)
        '''
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

    @classmethod
    def load(cls, train_loader, dev_loader, test_loader, instance_name, model_dir, checkpoint_epoch, nn_mode, device="cuda"):
        '''
        Returns an instance of Trainer by loading a previously saved model structure and checkpoint
        Args:
        - instance_name: instance name of the training session (also the name of the subdirectory containing the model)
        - model_dir: directory containing all models
        - checkpoint_epoch: training epoch to load
        OPTIONAL
        - dataset_path: path of the dataset (used when not in prediction mode)
        - device
        - prediction_mode: specifies that no gold data is going to be provided (used
            in the docker framework)
        - bert_model: provide an already built bert model
        - bert_tokenizer: provide an already built bert tokenizer
        Returns
        - an instance of Trainer containing the loaded training session
        ''' 
        
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

                #assert nn_mode == model.mode, "Required mode is :"+str(nn_mode)+" while model mode is: "+str(model.mode)
                
                model.to(device=device)

                print("Loading optimizer\n")
                optimizer = model_structure["optimizer"]
                print(optimizer)
                
                try:
                    lr_scheduler = model_structure["lr_scheduler"] 
                    print(lr_scheduler)
                except Exception as er:
                    lr_scheduler = None

                trainer = Trainer(instance_name, train_loader, dev_loader, test_loader, 
                            model, optimizer, device, nn_mode, model_dir, lr_scheduler = lr_scheduler)
                            
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

    def load_without_structure(self, checkpoint_epoch, path=None, device="cuda"):
        '''
        Loads only the model weights in the current training session (looking for \
        the subdirectory with name 'self.instance_name' of the current instance in 'path') 
        Args:
        - checkpoint_epoch: training epoch to load
        OPTIONAL
        - path: path of the directory containing all model subdirectories
        - device
        '''
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
    
    def get_gradient_stats(self, named_parameters):
        '''
        Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.
        
        Usage: Plug this function in Trainer class after loss.backwards() as 
        "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow
        '''
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

    #Taken from https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/8
    @classmethod
    def plot_grad_flow(gradient_magnitudes, show=False, save_to_dir=None):

        layers = [layer_name for layer_name, _ in gradient_magnitudes.items()]
        max_grads = [entry["max_grad"] for layer_name, entry in gradient_magnitudes.items()]
        #min_grads = [entry["min_grad"] for layer_name, entry in gradient_magnitudes.items()]
        avg_grads = [entry["avg_grad"] for layer_name, entry in gradient_magnitudes.items()]

        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
        plt.bar(np.arange(len(max_grads)), avg_grads, alpha=0.1, lw=1, color="b")
        plt.hlines(0, 0, len(avg_grads)+1, lw=2, color="k" )
        plt.xticks(range(0,len(avg_grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(avg_grads))
        plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
        plt.xlabel("Layers")
        plt.ylabel("Gradient magnitude")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.legend([matplotlib.lines.Line2D([0], [0], color="c", lw=4),
                    matplotlib.lines.Line2D([0], [0], color="b", lw=4),
                    matplotlib.lines.Line2D([0], [0], color="k", lw=4)], ['Max. gradient', 'Avg. gradient', 'Zero gradient'])
        
        if show:
            plt.show()

        if save_to_dir is not None:
            path = os.path.join(save_to_dir,"Gradient flow - Epoch "+str(self.last_epoch))+".png"
            plt.savefig(path)

class ResultsWriterThread(threading.Thread):
    '''Creates a small "preview" of the scores for a certain epoch
    '''
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


if __name__ == "__main__":

    INSTANCE_NAME = "PROVA"
    BATCH_SIZE = 128
    USE_CNN = True

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_dir = os.path.dirname(os.path.realpath(__file__))
    DATASET_DIR = os.path.join(base_dir,"data")
    DATASET_NAME = "UrbanSound8K"

    DATASET_PERCENTAGE = 1.0

    MODEL_DIR = os.path.join(base_dir,"model")

    selected_classes = [0,1,2,3,4,5,6,7,8,9]

    spectrogram_frames_per_segment = 128
    spectrogram_bands = 128
    in_channels = (3 if USE_CNN else 1)

    CNN_INPUT_SIZE = (spectrogram_bands, spectrogram_frames_per_segment, in_channels)
    FFN_INPUT_SIZE = 154

    right_shift_transformation = SpectrogramShift(input_size=CNN_INPUT_SIZE,width_shift_range=4,shift_prob=0.9)
    left_shift_transformation = SpectrogramShift(input_size=CNN_INPUT_SIZE,width_shift_range=4,shift_prob=0.9, left=True)
    random_side_shift_transformation = SpectrogramShift(input_size=CNN_INPUT_SIZE,width_shift_range=4,shift_prob=0.9, random_side=True)

    background_noise_transformation = SpectrogramAddGaussNoise(input_size=CNN_INPUT_SIZE,prob_to_have_noise=0.55)

    train_dataset = SoundDatasetFold(DATASET_DIR, DATASET_NAME, 
                                folds = [1], 
                                shuffle = True, 
                                generate_spectrograms = USE_CNN, 
                                shift_transformation = right_shift_transformation, 
                                background_noise_transformation = background_noise_transformation, 
                                audio_augmentation_pipeline = [], 
                                spectrogram_frames_per_segment = spectrogram_frames_per_segment, 
                                spectrogram_bands = spectrogram_bands, 
                                compute_deltas=True, 
                                compute_delta_deltas=True, 
                                test = False, 
                                progress_bar = True,
                                selected_classes=selected_classes,
                                select_percentage_of_dataset=DATASET_PERCENTAGE
                                )   

    test_dataset = SoundDatasetFold(DATASET_DIR, DATASET_NAME, 
                                folds = [2], 
                                shuffle = False, 
                                generate_spectrograms = USE_CNN, 
                                shift_transformation = right_shift_transformation, 
                                background_noise_transformation = background_noise_transformation, 
                                audio_augmentation_pipeline = [], 
                                spectrogram_frames_per_segment = spectrogram_frames_per_segment, 
                                spectrogram_bands = spectrogram_bands, 
                                compute_deltas=True, 
                                compute_delta_deltas=True, 
                                test = True, 
                                progress_bar = True,
                                selected_classes=selected_classes,
                                select_percentage_of_dataset=DATASET_PERCENTAGE
                                )


    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    if USE_CNN:
        model = ConvolutionalNetwork(CNN_INPUT_SIZE)
    else:
        model = FeedForwardNetwork(FFN_INPUT_SIZE, 256, train_dataset.get_num_classes())
    
    num_classes = train_dataset.get_num_classes()
    print("Number of classes: ", train_dataset.get_num_classes())
    print("Class names: ",train_dataset.class_distribution.keys())

    loss_function = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters())
    #optimizer = optim.SGD(model.parameters(), lr=0.001)
    #lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    trainer = Trainer(
                        INSTANCE_NAME,
                        BATCH_SIZE,
                        train_loader,
                        test_loader,
                        train_dataset.get_id_to_class(),
                        model,
                        loss_function,
                        optimizer,
                        DEVICE,
                        MODEL_DIR,
                        lr_scheduler=None,
                        cnn = USE_CNN
                    )
    
    trainer.train(30, save_test_scores_every=1, save_train_scores_every=1, save_model_every=1, compute_gradient_statistics=True)
