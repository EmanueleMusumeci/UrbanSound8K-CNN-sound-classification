'''
Generates all plots and graphical renders needed for the presentation
'''

if __name__ == "__main__":
        import os
        import math
        import random

        import dill

        import matplotlib.pyplot as plt

        from sklearn import metrics
        from sklearn.metrics import confusion_matrix

        import numpy as np

        from tqdm import tqdm

        from utils.plot_utils import *

        model_dir = "model"

        plot_color = "blue"
        
        SINGLE_PLOTS = True
        SINGLE_TRAIN_TEST_PLOTS = True
        CONFUSION_MATRIX = False
        COMPARATIVE_PLOTS = False
        GRADIENT_FLOW = False
        BEST_SCORES = False
        IMAGE_PREPROCESSING = False
        SALIENCY_MAPS = False

        ################
        # SINGLE PLOTS #
        ################
        if SINGLE_PLOTS:
                #Change the xticks_step to avoid the overlapping of labels on the x axis of graphs
                #Change the from/to_epoch and the epochs_skip to decide which score files are read
                #Use the combine tasks flag to plot a comparative plot of the same metric for all tasks
                """
                plot_scores("Base", model_dir, 
                                tasks={"audio_classification":"Audio classification"},
                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                from_epoch=0, to_epoch=49, epochs_skip=0, save_to_file=True,
                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                title_prefix = "Base",
                                color = plot_color
                                )
                plot_scores("PitchShift", model_dir,
                                tasks={"audio_classification":"Audio classification"},
                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                from_epoch=0, to_epoch=49, epochs_skip=0, save_to_file=True,
                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                title_prefix = "PitchShift",
                                color = plot_color
                                )
                """
                plot_scores("TimeStretch", model_dir,
                                tasks={"audio_classification":"Audio classification"},
                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                from_epoch=0, to_epoch=49, epochs_skip=0, save_to_file=True,
                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                title_prefix = "TimeStretch",
                                color = plot_color
                                )
                """
                plot_scores("BackgroundNoise", model_dir, 
                                tasks={"audio_classification":"Audio classification"},
                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                from_epoch=0, to_epoch=49, epochs_skip=0, save_to_file=True,
                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                title_prefix = "BackgroundNoise",
                                color = plot_color
                                )
                plot_scores("DynamicRangeCompression", model_dir, 
                                tasks={"audio_classification":"Audio classification"},
                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                from_epoch=0, to_epoch=49, epochs_skip=0, save_to_file=True,
                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                title_prefix = "DynamicRangeCompression",
                                color = plot_color
                                )
                """
        ###########################
        # SINGLE TRAIN/TEST PLOTS #
        ###########################
        if SINGLE_TRAIN_TEST_PLOTS:
                scores_dirs = {
                                "Test" : "scores_on_test",
                                "Train" : "scores_on_train"
                                }

                colors = {
                                "Test" : "blue",
                                "Train" : "orange"
                        }
                """
                plot_scores_from_multiple_dirs("Base", model_dir, 
                                                scores_dirs, tasks={"audio_classification":"Audio classification"},
                                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                                from_epoch=0, to_epoch=49, epochs_skip=0, save_to_file=True,
                                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                                title_prefix = "Base",
                                                colors = colors
                                                )
                
                plot_scores_from_multiple_dirs("PitchShift", model_dir, 
                                                scores_dirs, tasks={"audio_classification":"Audio classification"},
                                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                                from_epoch=0, to_epoch=49, epochs_skip=0, save_to_file=True,
                                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                                title_prefix = "PitchShift",
                                                colors = colors
                                                )
                """
                plot_scores_from_multiple_dirs("TimeStretch", model_dir, 
                                                scores_dirs, tasks={"audio_classification":"Audio classification"},
                                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                                from_epoch=0, to_epoch=49, epochs_skip=0, save_to_file=True,
                                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                                title_prefix = "TimeStretch",
                                                colors = colors
                                                )
                """
                plot_scores_from_multiple_dirs("BackgroundNoise", model_dir, 
                                                scores_dirs, tasks={"audio_classification":"Audio classification"},
                                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                                from_epoch=0, to_epoch=49, epochs_skip=0, save_to_file=True,
                                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                                title_prefix = "BackgroundNoise",
                                                colors = colors
                                                )
                
                plot_scores_from_multiple_dirs("DynamicRangeCompression", model_dir, 
                                                scores_dirs, tasks={"audio_classification":"Audio classification"},
                                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                                from_epoch=0, to_epoch=49, epochs_skip=0, save_to_file=True,
                                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                                title_prefix = "DynamicRangeCompression",
                                                colors = colors
                                                )
                """
        ####################
        # CONFUSION MATRIX #
        ####################
        if CONFUSION_MATRIX:
                plot_confusion_matrix("Base", model_dir, 
                                        tasks={"audio_classification":"Audio classification"},
                                        save_to_file=True, 
                                        title_prefix = "Base",
                                        scores_on_train=False
                                        )
                
                plot_confusion_matrix("PitchShift_PS2", model_dir, 
                                        tasks={"audio_classification":"Audio classification"},
                                        save_to_file=True, 
                                        title_prefix = "PitchShift",
                                        scores_on_train=False
                                        )
                
        #####################
        # Comparative plots #
        #####################
        if COMPARATIVE_PLOTS:
                model_names = {
                        "Base" : "Custom model - No augmentation", 
                        "PitchShift" : "Custom model - PitchShift",
                        }
                comparative_plots(model_names, model_dir,
                                tasks={"audio_classification":"Audio classification"},
                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                from_epoch=0, to_epoch=39, epochs_skip=0, save_to_file=True,
                                xticks_step=3, increase_epoch_labels_by_one=True,
                                title_prefix = "Custom model"
                                )  
        

        #################
        # GRADIENT FLOW #
        #################
        if GRADIENT_FLOW:
                #cropX = (0,250)
                cropX = None
                #cropY = (10,432)
                cropY = None
                create_gradient_flow_gif("Base", model_dir, cropX=cropX, cropY=cropY)

        ################
        # BEST  SCORES #
        ################
        if BEST_SCORES:
                model_names = {
                        "Custom classifier 16 layers - Normal",
                        }

                with open(os.path.join(model_dir,"best_scores.txt"), "w") as f:
                        for model_name in model_names:
                                _, best_scores_str = print_best_epoch_scores(model_name, model_dir,
                                        metrics = {"accuracy": "Accuracy",
                                                "precision": "Precision",
                                                "recall": "Recall",
                                                "f1": "F1 Macro Avg.",
                                                "distribution": "Distribution", 
                                                "confusion matrix": "Confusion Matrix",
                                                }
                                                )
                                f.write(best_scores_str)
        
        '''
        images = []

        images.append((
                load_image(os.path.join("data","Black Rat snake","0e54a884c28f6781a5aa7bec768c3067.jpg")),
                0,
                "Black Rat snake"))
        images.append((
                load_image(os.path.join("data","Common Garter snake","2df3ab34c46139a4814c013ccb6218c8.jpg")),
                1,
                "Common Garter snake"))
        images.append((
                load_image(os.path.join("data","DeKay's Brown snake","a697c4a5205df2b5a32ff3538d08387f.jpg")),
                2,
                "DeKay's Brown snake"))
        images.append((
                load_image(os.path.join("data","Northern Watersnake","c9cadc328c4911c12558df424a15fb01.jpg")),
                3,
                "Northern Watersnake"))
        images.append((
                load_image(os.path.join("data","Western Diamondback rattlesnake","ca7b19ad649466877af003d13244db86.jpg")),
                4,
                "Western Diamondback rattlesnake"))
        '''
        
        ################################
        # IMAGE PREPROCESSING PIPELINE #
        ################################
        if IMAGE_PREPROCESSING:
                for image in images:
                        #Training preprocessing pipeline
                        show_preprocessing({
                                "1 - RandomHorizontalFlip" : transforms.RandomHorizontalFlip(p=1.0),
                                "2 - RandomVerticalFlip" : transforms.RandomVerticalFlip(p=1.0),
                                "3 - ColorJitter" : transforms.ColorJitter(brightness = 0.2, saturation=0.2),
                                "4 - RandomCrop" : transforms.RandomResizedCrop((224,224)),
                                }, image[0], title_prefix = "Training", save_to_dir=os.path.join(model_dir,"plots","preprocessed_image_examples",image[2]))

                        #Validation preprocessing pipeline
                        show_preprocessing({
                                "1 - CenterCrop" : transforms.CenterCrop((224,224)),
                                }, image[0], title_prefix = "Validation", save_to_dir=os.path.join(model_dir,"plots","preprocessed_image_examples",image[2]))

        #################
        # SALIENCY MAPS #
        #################
        if SALIENCY_MAPS:
                training_preprocessing_pipeline = transforms.Compose([
                        transforms.Resize((224,224)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])
                validation_preprocessing_pipeline = transforms.Compose([
                        transforms.Resize((224,224)),
                        ])

                for image in images:
                        #custom_model_wrapper = CustomTrainer.load("Custom classifier 20 layers - Final", 32, None, None, None, model_dir, 48)
                        #layer = 0
                        #visualize_features(custom_model_wrapper.model, image[0], image[1],
                        #                        training_preprocessing_pipeline, validation_preprocessing_pipeline, 
                        #                        save_to_dir=os.path.join(model_dir,"plots","saliency_maps"), filename="Feature_activation_layer_"+str(layer))
                        
                        custom_model_wrapper = CustomTrainer.load("Custom classifier 20 layers - Final", 32, None, None, None, model_dir, 48)
                        visualize_features_on_layers(custom_model_wrapper.model, image[0], image[1],
                                                training_preprocessing_pipeline, validation_preprocessing_pipeline,
                                                save_to_dir=os.path.join(model_dir,"plots","saliency_maps","Custom20",image[2]),
                                                from_layer = 0, to_layer=18,
                                                filename_prefix="Custom_classifier_20_Feature_activation_layer_")

