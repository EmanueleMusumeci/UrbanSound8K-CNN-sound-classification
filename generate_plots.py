'''
Generates all plots and graphical renders needed for the presentation
'''

if __name__ == "__main__":
        import os
        import math
        import random

        import shutils

        from shutil import copyfile, rmtree

        import dill

        import matplotlib.pyplot as plt

        from sklearn import metrics
        from sklearn.metrics import confusion_matrix

        import numpy as np

        from tqdm import tqdm

        from utils.plot_utils import *
        from utils.dataset_utils import *
        from utils.audio_utils import load_audio_file, play_sound, SingleWindowSelector
        from data_augmentation.audio_transformations import *
        from Trainer import *

        dataset_dir = "data"
        model_dir = "model"
        plot_dir = os.path.join("plots")

        plot_color = "blue"
        
        SINGLE_PLOTS = False
        SINGLE_TRAIN_TEST_PLOTS = False
        CONFUSION_MATRIX = False
        COMPARATIVE_PLOTS = False
        GRADIENT_FLOW = False
        BEST_SCORES = False
        PLOT_BEST_SCORES_DELTAS = True
        PLOT_CLASS_DISTRIBUTION = False
        PLOT_PREPROCESSING_ACCURACY_RESULTS = False
        COLLECT_AND_PREPROCESS_SAMPLES = False
        SHOW_PREPROCESSING = False
        SALIENCY_MAPS = False

        ################
        # SINGLE PLOTS 
        ################
        if SINGLE_PLOTS:
                #Change the xticks_step to avoid the overlapping of labels on the x axis of graphs
                #Change the from/to_epoch and the epochs_skip to decide which score files are read
                #Use the combine tasks flag to plot a comparative plot of the same metric for all tasks
                plot_scores("Base", model_dir, 
                                tasks={"audio_classification":"Audio classification"},
                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                from_epoch=0, to_epoch=49, epochs_skip=0,
                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                title_prefix = "Base",
                                color = plot_color,
                                plot_dir = plot_dir
                                )

                plot_scores("PitchShift", model_dir,
                                tasks={"audio_classification":"Audio classification"},
                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                from_epoch=0, to_epoch=49, epochs_skip=0, 
                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                title_prefix = "PitchShift",
                                color = plot_color,
                                plot_dir = plot_dir
                                )
                '''
                plot_scores("TimeStretch", model_dir,
                                tasks={"audio_classification":"Audio classification"},
                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                from_epoch=0, to_epoch=49, epochs_skip=0, 
                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                title_prefix = "TimeStretch",
                                color = plot_color,
                                plot_dir = plot_dir
                                )
                
                plot_scores("BackgroundNoise", model_dir, 
                                tasks={"audio_classification":"Audio classification"},
                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                from_epoch=0, to_epoch=49, epochs_skip=0, 
                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                title_prefix = "BackgroundNoise",
                                color = plot_color,
                                plot_dir = plot_dir
                                )
                
                plot_scores("DynamicRangeCompression", model_dir, 
                                tasks={"audio_classification":"Audio classification"},
                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                from_epoch=0, to_epoch=49, epochs_skip=0, 
                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                title_prefix = "DynamicRangeCompression",
                                color = plot_color,
                                plot_dir = plot_dir
                                )
                '''

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

                plot_scores_from_multiple_dirs("Base", model_dir, 
                                                scores_dirs, tasks={"audio_classification":"Audio classification"},
                                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                                from_epoch=0, to_epoch=49, epochs_skip=0, 
                                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                                title_prefix = "Base",
                                                colors = colors,
                                                plot_dir = plot_dir
                                                )

                plot_scores_from_multiple_dirs("PitchShift", model_dir, 
                                                scores_dirs, tasks={"audio_classification":"Audio classification"},
                                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                                from_epoch=0, to_epoch=49, epochs_skip=0, 
                                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                                title_prefix = "PitchShift",
                                                colors = colors,
                                                plot_dir = plot_dir
                                                )
                '''
                plot_scores_from_multiple_dirs("TimeStretch", model_dir, 
                                                scores_dirs, tasks={"audio_classification":"Audio classification"},
                                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                                from_epoch=0, to_epoch=49, epochs_skip=0, 
                                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                                title_prefix = "TimeStretch",
                                                colors = colors,
                                                plot_dir = plot_dir
                                                )
                
                plot_scores_from_multiple_dirs("BackgroundNoise", model_dir, 
                                                scores_dirs, tasks={"audio_classification":"Audio classification"},
                                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                                from_epoch=0, to_epoch=49, epochs_skip=0, 
                                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                                title_prefix = "BackgroundNoise",
                                                colors = colors,
                                                plot_dir = plot_dir
                                                )

                plot_scores_from_multiple_dirs("DynamicRangeCompression", model_dir, 
                                                scores_dirs, tasks={"audio_classification":"Audio classification"},
                                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                                from_epoch=0, to_epoch=49, epochs_skip=0, 
                                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                                title_prefix = "DynamicRangeCompression",
                                                colors = colors,
                                                plot_dir = plot_dir
                                                )
                '''

        ####################
        # CONFUSION MATRIX #
        ####################
        if CONFUSION_MATRIX:
                plot_confusion_matrix("Base", model_dir, 
                                        tasks={"audio_classification":"Audio classification"},
                                         
                                        title_prefix = "Base",
                                        scores_on_train=False,
                                        plot_dir = plot_dir
                                        )
                
                '''
                plot_confusion_matrix("PitchShift_PS2", model_dir, 
                                        tasks={"audio_classification":"Audio classification"},
                                         
                                        title_prefix = "PitchShift",
                                        scores_on_train=False,
                                        plot_dir = plot_dir
                                        )
                '''

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
                                from_epoch=0, to_epoch=39, epochs_skip=0, 
                                xticks_step=3, increase_epoch_labels_by_one=True,
                                title_prefix = "Custom model",
                                plot_dir = plot_dir
                                )  
        

        #################
        # GRADIENT FLOW #
        #################
        if GRADIENT_FLOW:
                #cropX = (0,250)
                cropX = None
                #cropY = (10,432)
                cropY = None
                create_gradient_flow_gif("Base", model_dir, plot_dir, cropX=cropX, cropY=cropY)

        ################
        # BEST  SCORES #
        ################
        if BEST_SCORES:
                model_names = {
                        "Base",
                        }

                with open(os.path.join(model_dir,"best_scores.txt"), "w") as f:
                        for model_name in model_names:
                                _, best_scores_str = get_best_epoch_scores(model_name, model_dir,
                                        metrics = {"accuracy": "Accuracy",
                                                "precision": "Precision",
                                                "recall": "Recall",
                                                "f1": "F1 Macro Avg.",
                                                "distribution": "Distribution", 
                                                "confusion matrix": "Confusion Matrix",
                                                }
                                                )
                                f.write(best_scores_str)
        
        ################
        # BEST  SCORES #
        ################
        if PLOT_BEST_SCORES_DELTAS:
                
                model_names = {
                        "Base" : ["BackgroundNoise", "DynamicRangeCompression"]
                        }

                plot_model_improvement_deltas(model_names, model_dir, 
                                                tasks={"audio_classification" : "Audio classification"},
                                                metrics={"Accuracy":["accuracy"]}, 
                                                epoch=0, to_epoch=0, epochs_skip=0,
                                                scores_on_train=False,
                                                title_prefix=None,
                                                plot_dir = None, 
                                                show = False
                                             )
        #############################
        #  PLOT CLASS DISTRIBUTION  #
        #############################
        if PLOT_CLASS_DISTRIBUTION:
                audio_meta, audio_raw, audio_spectrograms_raw = load_raw_compacted_dataset(dataset_dir, folds = [1,2,3,4,5,6,7,8,9,10])
                class_distribution = {}
                for sample_meta in audio_meta:
                        if sample_meta["class_name"] not in class_distribution:
                                class_distribution[sample_meta["class_name"]] = 0
                        else:
                                class_distribution[sample_meta["class_name"]] += 1
                plot_class_distributions(class_distribution, plot_dir = plot_dir)

        ###############################
        #  PLOT RESULTS DISTRIBUTION  #
        ###############################
        if PLOT_PREPROCESSING_ACCURACY_RESULTS:
                pass

#TODO: Integrare lavoro Michele

        ####################################
        #  COLLECT AND PREPROCESS SAMPLES  #
        ####################################
        #Collect one sample for each class
        selected_classes = [1,2,3,4,5,6,7,8,9,10]
        #selected_classes = [1]
        if SHOW_PREPROCESSING or SALIENCY_MAPS:
                fold_meta, fold_raw, fold_spectrograms_raw = load_raw_compacted_dataset(dataset_dir, folds = [1])
                samples_one_for_each_class = {}
                samples_dir = os.path.join(plot_dir, "preprocessing", "chosen_samples")
                if os.path.exists(os.path.join(plot_dir, "preprocessing")) and SHOW_PREPROCESSING:
                        shutil.rmtree(os.path.join(plot_dir, "preprocessing"))
                        os.makedirs(os.path.join(plot_dir, "preprocessing"))
                        

                for sound_class in selected_classes:
                        for i, sample_meta in enumerate(fold_meta):
                                if sample_meta["class_id"] == sound_class:
                                        sample_dir = os.path.join(samples_dir, sample_meta["class_name"])
                                        if not os.path.exists(sample_dir):
                                                os.makedirs(sample_dir)

                                        if np.random.rand()>0.3:
                                                continue
                                        path_separator = os.path.sep
                                        sample_file_name = sample_meta["file_path"].split(path_separator)[-1]

                                        samples_one_for_each_class[sample_meta["class_name"]] = {
                                                                        "audio":fold_raw[i], 
                                                                        "file_name":sample_file_name, 
                                                                        "class_name":sample_meta["class_name"], 
                                                                        "class_id":sample_meta["class_id"], 
                                                                        "sample_rate":sample_meta["sampling_rate"],
                                                                        "sample_meta":sample_meta
                                                                        }
                                        break

                print(samples_one_for_each_class)
                
                #Preprocess the collected samples
                preprocessors = [
                                #PitchShift(values = [-2, -1, 1, 2])
                                PitchShift(values = [-3.5]),
                                MUDADynamicRangeCompression(),
                                #BackgroundNoise({
                                #"street_scene_1" : "150993__saphe__street-scene-1.wav",
                                #    "street_scene_3" : "173955__saphe__street-scene-3.wav",
                                #    "street_valencia" : "207208__jormarp__high-street-of-gandia-valencia-spain.wav",
                                #    "city_park_tel_aviv" : "268903__yonts__city-park-tel-aviv-israel.wav",
                                #}, files_dir = os.path.join(dataset_dir, "UrbanSound8K-JAMS", "background_noise")),
                                TimeStretch(values = [0.5])
                                ]

                for class_name, sample in samples_one_for_each_class.items():
                        #clip, sr = load_audio_file(os.path.join(samples_dir,filename))
                        clip = sample["audio"]
                        sr = sample["sample_rate"]
                        class_name = sample["class_name"]
                        file_name = sample["file_name"]
                        if not os.path.exists(os.path.join(samples_dir, class_name)):
                                os.makedirs(os.path.join(samples_dir, class_name))
                        scipy.io.wavfile.write(os.path.join(samples_dir,class_name,file_name), sr, clip)

        ################################
        #  SHOW PREPROCESSING PREVIEW  #
        ################################
        if SHOW_PREPROCESSING:
                for class_name, sample in samples_one_for_each_class.items():        
                        show_audio_preprocessing(sample["audio"], preprocessors, \
                                                save_clips_to_dir = os.path.join(plot_dir, "preprocessing", "chosen_samples"), \
                                                save_plots_to_dir = os.path.join(plot_dir, "preprocessing"), \
                                                horizontal = True, \
                                                show = False, \
                                                sound_file_name=sample["file_name"], sound_class=sample["class_name"])
        
        #################
        # SALIENCY MAPS #
        #################
        if SALIENCY_MAPS:
                if os.path.exists(os.path.join(plot_dir, "saliency_maps")):
                        shutil.rmtree(os.path.join(plot_dir, "saliency_maps"))
                        os.makedirs(os.path.join(plot_dir, "saliency_maps"))
                else:
                        os.makedirs(os.path.join(plot_dir, "saliency_maps"))

                '''
                training_preprocessing_pipeline = transforms.Compose([
                        transforms.Resize((224,224)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])
                validation_preprocessing_pipeline = transforms.Compose([
                        transforms.Resize((224,224)),
                        ])
                
                for image in images:
                        #custom_model_wrapper = Trainer.load("Custom classifier 20 layers - Final", 32, None, None, None, model_dir, 48)
                        #layer = 0
                        #visualize_features(custom_model_wrapper.model, image[0], image[1],
                        #                        training_preprocessing_pipeline, validation_preprocessing_pipeline, 
                        #                        save_to_dir=os.path.join(model_dir,"plots","saliency_maps"), filename="Feature_activation_layer_"+str(layer))
                        
                        custom_model_wrapper = Trainer.load("Base", 32, None, None, None, model_dir, 48)
                        visualize_features_on_layers(custom_model_wrapper.model, image[0], image[1],
                                                training_preprocessing_pipeline, validation_preprocessing_pipeline,
                                                save_to_dir=os.path.join(plot_dir,"saliency_maps","Custom20",image[2]),
                                                from_layer = 0, to_layer=18,
                                                filename_prefix="Custom_classifier_20_Feature_activation_layer_")
                '''
                models = {
                                "PROVA" : {"epoch" : 2, "preprocessor" : None, "deltas" : False, "delta_deltas" : False, "to_layer" : 3},
                         }

                CLIP_SECONDS = 3
                SPECTROGRAM_HOP_LENGTH = 512
                SAMPLE_RATE = 22050
                spectrogram_frames_per_segment = CLIP_SECONDS*SAMPLE_RATE / SPECTROGRAM_HOP_LENGTH
                
                for model_name, model_entry in models.items():
                        for class_name, sample in samples_one_for_each_class.items():
                  
                                dataset = SoundDatasetFold(dataset_dir, "", 
                                        folds = [], 
                                        preprocessor = model_entry["preprocessor"],
                                        use_spectrograms = True, 
                                        #image_shift_transformation = right_shift_transformation, 
                                        #image_background_noise_transformation = background_noise_transformation, 

                                        spectrogram_frames_per_segment = spectrogram_frames_per_segment, 
                                        spectrogram_bands = 128, 
                                        compute_deltas=model_entry["deltas"], 
                                        compute_delta_deltas=model_entry["delta_deltas"], 
                                        selected_classes=selected_classes,
                                        audio_segment_selector=SingleWindowSelector(CLIP_SECONDS, spectrogram_hop_length=SPECTROGRAM_HOP_LENGTH, random_location = False),
                                        debug_preprocessing=True
                                ) 

                                data_loader = DataLoader(dataset)
                                
                                custom_model_wrapper = Trainer.load(data_loader, None, model_name, model_dir, model_entry["epoch"])                      
                                
                                #Set the model to evaluation mode to prevent dropout
                                custom_model_wrapper.model.eval()

                                preprocessed_sample = data_loader(sample["sample_meta"])
                                
                                output_dir = os.path.join(plot_dir,"saliency_maps",model_name,sample["class_name"])
                                if not os.path.exists(output_dir):
                                        os.makedirs(output_dir)
                                
                                if model_name.endswith("custom_model"):
                                        pass
                                else:
                                        convolutional_layers = custom_model_wrapper.model.convolutional_layers 
                                        dense_layers = custom_model_wrapper.model.dense_layers 

                                visualize_features_on_layers(custom_model_wrapper.model, convolutional_layers, dense_layers, 
                                                                preprocessed_sample, sample["class_id"],
                                                                save_to_dir=output_dir,
                                                                from_layer = 0, to_layer=model_entry["to_layer"],
                                                                filename_prefix=model_name+"_layer_")
                