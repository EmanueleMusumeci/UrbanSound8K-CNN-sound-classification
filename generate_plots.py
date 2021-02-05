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
        PREPROCESSING_PERFORMANCE_DELTA_COMPARISONS = True
        PLOT_TRAIN_TEST_ACCURACY_DELTAS = True 
        PLOT_CLASS_DISTRIBUTION = False
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
                
                #TODO: All models
                plot_scores("Base", model_dir, 
                                tasks={"audio_classification":"Audio classification"},
                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                from_epoch=0, to_epoch=49, epochs_skip=0,
                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                title_prefix = "Paper model",
                                color = plot_color,
                                plot_dir = plot_dir
                                )

                plot_scores("Base_custom", model_dir, 
                                tasks={"audio_classification":"Audio classification"},
                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                from_epoch=0, to_epoch=49, epochs_skip=0,
                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                title_prefix = "Custom model",
                                color = plot_color,
                                plot_dir = plot_dir
                                )
                
                plot_scores("BackgroundNoise", model_dir, 
                                tasks={"audio_classification":"Audio classification"},
                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                from_epoch=0, to_epoch=49, epochs_skip=0,
                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                title_prefix = "Paper model - Background Noise",
                                color = plot_color,
                                plot_dir = plot_dir
                                )

                plot_scores("BackgroundNoise_custom", model_dir, 
                                tasks={"audio_classification":"Audio classification"},
                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                from_epoch=0, to_epoch=49, epochs_skip=0,
                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                title_prefix = "Custom model - Background Noise",
                                color = plot_color,
                                plot_dir = plot_dir
                                )
                
                plot_scores("DynamicRangeCompression", model_dir, 
                                tasks={"audio_classification":"Audio classification"},
                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                from_epoch=0, to_epoch=49, epochs_skip=0,
                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                title_prefix = "Paper model - Dynamic Range Compression",
                                color = plot_color,
                                plot_dir = plot_dir
                                )

                plot_scores("DynamicRangeCompression_custom", model_dir, 
                                tasks={"audio_classification":"Audio classification"},
                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                from_epoch=0, to_epoch=49, epochs_skip=0,
                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                title_prefix = "Custom model - Dynamic Range Compression",
                                color = plot_color,
                                plot_dir = plot_dir
                                )
                
                plot_scores("PitchShift_PS1", model_dir, 
                                tasks={"audio_classification":"Audio classification"},
                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                from_epoch=0, to_epoch=49, epochs_skip=0,
                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                title_prefix = "Paper model - Pitch Shift (shorter range)",
                                color = plot_color,
                                plot_dir = plot_dir
                                )

                plot_scores("PitchShift_PS1_custom", model_dir, 
                                tasks={"audio_classification":"Audio classification"},
                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                from_epoch=0, to_epoch=49, epochs_skip=0,
                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                title_prefix = "Custom model - Pitch Shift (shorter range)",
                                color = plot_color,
                                plot_dir = plot_dir
                                )

                plot_scores("PitchShift_PS2", model_dir, 
                                tasks={"audio_classification":"Audio classification"},
                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                from_epoch=0, to_epoch=49, epochs_skip=0,
                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                title_prefix = "Paper model - Pitch Shift (wider range)",
                                color = plot_color,
                                plot_dir = plot_dir
                                )

                plot_scores("PitchShift_PS2_custom", model_dir, 
                                tasks={"audio_classification":"Audio classification"},
                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                from_epoch=0, to_epoch=49, epochs_skip=0,
                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                title_prefix = "Custom model - Pitch Shift (wider range)",
                                color = plot_color,
                                plot_dir = plot_dir
                                )
                
                plot_scores("TimeStretch", model_dir, 
                                tasks={"audio_classification":"Audio classification"},
                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                from_epoch=0, to_epoch=49, epochs_skip=0,
                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                title_prefix = "Paper model - Time Stretch",
                                color = plot_color,
                                plot_dir = plot_dir
                                )

                plot_scores("TimeStretch_custom", model_dir, 
                                tasks={"audio_classification":"Audio classification"},
                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                from_epoch=0, to_epoch=49, epochs_skip=0,
                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                title_prefix = "Custom model - Time Stretch",
                                color = plot_color,
                                plot_dir = plot_dir
                                )
                
                plot_scores("Base_delta", model_dir, 
                                tasks={"audio_classification":"Audio classification"},
                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                from_epoch=0, to_epoch=49, epochs_skip=0,
                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                title_prefix = "Paper model with delta features",
                                color = plot_color,
                                plot_dir = plot_dir
                                )
                
                plot_scores("BackgroundNoise_delta", model_dir, 
                                tasks={"audio_classification":"Audio classification"},
                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                from_epoch=0, to_epoch=49, epochs_skip=0,
                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                title_prefix = "Paper model with Background Noise and delta features",
                                color = plot_color,
                                plot_dir = plot_dir
                                )

                plot_scores("DynamicRangeCompression_delta", model_dir, 
                                tasks={"audio_classification":"Audio classification"},
                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                from_epoch=0, to_epoch=49, epochs_skip=0,
                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                title_prefix = "Paper model with Dynamic Range Compression and delta features",
                                color = plot_color,
                                plot_dir = plot_dir
                                )
                
                plot_scores("PitchShift_PS1_delta", model_dir, 
                                tasks={"audio_classification":"Audio classification"},
                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                from_epoch=0, to_epoch=49, epochs_skip=0,
                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                title_prefix = "Paper model with Pitch Shift (shorter range) and delta features",
                                color = plot_color,
                                plot_dir = plot_dir
                                )
                
                plot_scores("PitchShift_PS2_delta", model_dir, 
                                tasks={"audio_classification":"Audio classification"},
                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                from_epoch=0, to_epoch=49, epochs_skip=0,
                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                title_prefix = "Paper model with Pitch Shift (longer range) and delta features",
                                color = plot_color,
                                plot_dir = plot_dir
                                )
                
                plot_scores("TimeStretch_delta", model_dir, 
                                tasks={"audio_classification":"Audio classification"},
                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                from_epoch=0, to_epoch=49, epochs_skip=0,
                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                title_prefix = "Paper model with Time Stretch and delta features",
                                color = plot_color,
                                plot_dir = plot_dir
                                )

                plot_scores("Base_delta_delta", model_dir, 
                                tasks={"audio_classification":"Audio classification"},
                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                from_epoch=0, to_epoch=49, epochs_skip=0,
                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                title_prefix = "Paper model with delta-delta features",
                                color = plot_color,
                                plot_dir = plot_dir
                                )
                
                plot_scores("BackgroundNoise_delta_delta", model_dir, 
                                tasks={"audio_classification":"Audio classification"},
                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                from_epoch=0, to_epoch=49, epochs_skip=0,
                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                title_prefix = "Paper model with Background Noise and delta-delta features",
                                color = plot_color,
                                plot_dir = plot_dir
                                )

                plot_scores("DynamicRangeCompression_delta_delta", model_dir, 
                                tasks={"audio_classification":"Audio classification"},
                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                from_epoch=0, to_epoch=49, epochs_skip=0,
                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                title_prefix = "Paper model with Dynamic Range Compression and delta-delta features",
                                color = plot_color,
                                plot_dir = plot_dir
                                )
                
                plot_scores("PitchShift_PS1_delta_delta", model_dir, 
                                tasks={"audio_classification":"Audio classification"},
                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                from_epoch=0, to_epoch=49, epochs_skip=0,
                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                title_prefix = "Paper model with Pitch Shift (shorter range) and delta-delta features",
                                color = plot_color,
                                plot_dir = plot_dir
                                )
                
                plot_scores("PitchShift_PS2_delta_delta", model_dir, 
                                tasks={"audio_classification":"Audio classification"},
                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                from_epoch=0, to_epoch=49, epochs_skip=0,
                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                title_prefix = "Paper model with Pitch Shift (longer range) and delta-delta features",
                                color = plot_color,
                                plot_dir = plot_dir
                            )
                
                plot_scores("TimeStretch_delta_delta", model_dir, 
                                tasks={"audio_classification":"Audio classification"},
                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                from_epoch=0, to_epoch=49, epochs_skip=0,
                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                title_prefix = "Paper model with Time Stretch and delta-delta features",
                                color = plot_color,
                                plot_dir = plot_dir
                                )
                
                plot_scores("Base_IMAGE_SHIFT_custom", model_dir, 
                                tasks={"audio_classification":"Audio classification"},
                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                from_epoch=0, to_epoch=49, epochs_skip=0,
                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                title_prefix = "Custom model - Random spectrogram image right shift augmentation",
                                color = plot_color,
                                plot_dir = plot_dir
                                )
                
                plot_scores("Base_IMAGE_NOISE_custom", model_dir, 
                                tasks={"audio_classification":"Audio classification"},
                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                from_epoch=0, to_epoch=49, epochs_skip=0,
                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                title_prefix = "Custom model - Random spectrogram image noise augmentation",
                                color = plot_color,
                                plot_dir = plot_dir
                                )
                
                plot_scores("Base_IMAGE_SHIFT_NOISE_custom", model_dir, 
                                tasks={"audio_classification":"Audio classification"},
                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                from_epoch=0, to_epoch=49, epochs_skip=0,
                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                title_prefix = "Custom model - Random spectrogram image right shift and noise augmentation",
                                color = plot_color,
                                plot_dir = plot_dir
                                )
                
                plot_scores("Base_IMAGE_SHIFT", model_dir, 
                                tasks={"audio_classification":"Audio classification"},
                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                from_epoch=0, to_epoch=49, epochs_skip=0,
                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                title_prefix = " Paper model - Random spectrogram image right shift augmentation",
                                color = plot_color,
                                plot_dir = plot_dir
                                )
                
                plot_scores("Base_IMAGE_NOISE", model_dir, 
                                tasks={"audio_classification":"Audio classification"},
                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                from_epoch=0, to_epoch=49, epochs_skip=0,
                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                title_prefix = "Paper model - Random spectrogram image noise augmentation",
                                color = plot_color,
                                plot_dir = plot_dir
                                )
                
                plot_scores("Base_IMAGE_SHIFT_NOISE", model_dir, 
                                tasks={"audio_classification":"Audio classification"},
                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                from_epoch=0, to_epoch=49, epochs_skip=0,
                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                title_prefix = "Paper model - Image Random Noise and Shift Augmentation",
                                color = plot_color,
                                plot_dir = plot_dir
                                )
                
                            
                
                

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
                                                title_prefix = "Paper model ",
                                                colors = colors,
                                                plot_dir = plot_dir
                                              )
                
                plot_scores_from_multiple_dirs("Base_custom", model_dir, 
                                                scores_dirs, tasks={"audio_classification":"Audio classification"},
                                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                                from_epoch=0, to_epoch=49, epochs_skip=0, 
                                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                                title_prefix = "Custom model",
                                                colors = colors,
                                                plot_dir = plot_dir
                                              )
                
                plot_scores_from_multiple_dirs("BackgroundNoise", model_dir, 
                                                scores_dirs, tasks={"audio_classification":"Audio classification"},
                                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                                from_epoch=0, to_epoch=49, epochs_skip=0, 
                                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                                title_prefix = "Paper model - Background Noise",
                                                colors = colors,
                                                plot_dir = plot_dir
                                              )
                
                plot_scores_from_multiple_dirs("BackgroundNoise_custom", model_dir, 
                                                scores_dirs, tasks={"audio_classification":"Audio classification"},
                                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                                from_epoch=0, to_epoch=49, epochs_skip=0, 
                                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                                title_prefix = "Custom model - Background Noise",
                                                colors = colors,
                                                plot_dir = plot_dir
                                              )
                
                plot_scores_from_multiple_dirs("DynamicRangeCompression", model_dir, 
                                                scores_dirs, tasks={"audio_classification":"Audio classification"},
                                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                                from_epoch=0, to_epoch=49, epochs_skip=0, 
                                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                                title_prefix = "Paper model - Dynamic Range Compression",
                                                colors = colors,
                                                plot_dir = plot_dir
                                                )
                
                plot_scores_from_multiple_dirs("DynamicRangeCompression_custom", model_dir, 
                                                scores_dirs, tasks={"audio_classification":"Audio classification"},
                                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                                from_epoch=0, to_epoch=49, epochs_skip=0, 
                                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                                title_prefix = "Custom model - Dynamic Range Compression",
                                                colors = colors,
                                                plot_dir = plot_dir
                                                )
                
                plot_scores_from_multiple_dirs("PitchShift_PS1", model_dir, 
                                                scores_dirs, tasks={"audio_classification":"Audio classification"},
                                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                                from_epoch=0, to_epoch=49, epochs_skip=0, 
                                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                                title_prefix = "Paper model - Pitch Shift (shorter range)",
                                                colors = colors,
                                                plot_dir = plot_dir
                                                )
                
                plot_scores_from_multiple_dirs("PitchShift_PS1_custom", model_dir, 
                                                scores_dirs, tasks={"audio_classification":"Audio classification"},
                                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                                from_epoch=0, to_epoch=49, epochs_skip=0, 
                                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                                title_prefix = "Custom model - Pitch Shift (shorter range)",
                                                colors = colors,
                                                plot_dir = plot_dir
                                                )
                
                plot_scores_from_multiple_dirs("PitchShift_PS2", model_dir, 
                                                scores_dirs, tasks={"audio_classification":"Audio classification"},
                                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                                from_epoch=0, to_epoch=49, epochs_skip=0, 
                                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                                title_prefix = "Paper model - Pitch Shift (wider range)",
                                                colors = colors,
                                                plot_dir = plot_dir
                                                )
                
                plot_scores_from_multiple_dirs("PitchShift_PS2_custom", model_dir, 
                                                scores_dirs, tasks={"audio_classification":"Audio classification"},
                                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                                from_epoch=0, to_epoch=49, epochs_skip=0, 
                                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                                title_prefix = "Custom model - Pitch Shift (wider range)",
                                                colors = colors,
                                                plot_dir = plot_dir
                                                )
                
                plot_scores_from_multiple_dirs("TimeStretch", model_dir, 
                                                scores_dirs, tasks={"audio_classification":"Audio classification"},
                                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                                from_epoch=0, to_epoch=49, epochs_skip=0, 
                                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                                title_prefix = "Paper model - Time Stretch",
                                                colors = colors,
                                                plot_dir = plot_dir
                                                )
                
                plot_scores_from_multiple_dirs("TimeStretch_custom", model_dir, 
                                                scores_dirs, tasks={"audio_classification":"Audio classification"},
                                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                                from_epoch=0, to_epoch=49, epochs_skip=0, 
                                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                                title_prefix = "Custom model - Time Stretch",
                                                colors = colors,
                                                plot_dir = plot_dir
                                                )
                
                plot_scores_from_multiple_dirs("Base_delta", model_dir, 
                                                scores_dirs, tasks={"audio_classification":"Audio classification"},
                                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                                from_epoch=0, to_epoch=49, epochs_skip=0, 
                                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                                title_prefix = "Paper model with delta features",
                                                colors = colors,
                                                plot_dir = plot_dir
                                                )
                
                plot_scores_from_multiple_dirs("BackgroundNoise_delta", model_dir, 
                                                scores_dirs, tasks={"audio_classification":"Audio classification"},
                                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                                from_epoch=0, to_epoch=49, epochs_skip=0, 
                                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                                title_prefix = "Paper model with Background Noise and delta features",
                                                colors = colors,
                                                plot_dir = plot_dir
                                                )
                
                plot_scores_from_multiple_dirs("DynamicRangeCompression_delta", model_dir, 
                                                scores_dirs, tasks={"audio_classification":"Audio classification"},
                                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                                from_epoch=0, to_epoch=49, epochs_skip=0, 
                                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                                title_prefix = "Paper model with Dynamic Range Compression and delta features",
                                                colors = colors,
                                                plot_dir = plot_dir
                                                )
                
                plot_scores_from_multiple_dirs("PitchShift_PS1_delta", model_dir, 
                                                scores_dirs, tasks={"audio_classification":"Audio classification"},
                                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                                from_epoch=0, to_epoch=49, epochs_skip=0, 
                                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                                title_prefix = "Paper model with Pitch Shift (shorter range) and delta-delta features",
                                                colors = colors,
                                                plot_dir = plot_dir
                                                )
                
                plot_scores_from_multiple_dirs("PitchShift_PS2_delta", model_dir, 
                                                scores_dirs, tasks={"audio_classification":"Audio classification"},
                                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                                from_epoch=0, to_epoch=49, epochs_skip=0, 
                                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                                title_prefix = "Paper model with Pitch Shift (longer range) and delta-delta features",
                                                colors = colors,
                                                plot_dir = plot_dir
                                                )
                
                plot_scores_from_multiple_dirs("TimeStretch_delta", model_dir, 
                                                scores_dirs, tasks={"audio_classification":"Audio classification"},
                                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                                from_epoch=0, to_epoch=49, epochs_skip=0, 
                                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                                title_prefix = "Paper model with Time Stretch and delta-delta features",
                                                colors = colors,
                                                plot_dir = plot_dir
                                                )
                
                plot_scores_from_multiple_dirs("Base_delta_delta", model_dir, 
                                                scores_dirs, tasks={"audio_classification":"Audio classification"},
                                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                                from_epoch=0, to_epoch=49, epochs_skip=0, 
                                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                                title_prefix = "Paper model with delta-delta features",
                                                colors = colors,
                                                plot_dir = plot_dir
                                                )
                
                plot_scores_from_multiple_dirs("BackgroundNoise_delta_delta", model_dir, 
                                                scores_dirs, tasks={"audio_classification":"Audio classification"},
                                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                                from_epoch=0, to_epoch=49, epochs_skip=0, 
                                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                                title_prefix = "Paper model with Background Noise and delta-delta features",
                                                colors = colors,
                                                plot_dir = plot_dir
                                                )
                
                plot_scores_from_multiple_dirs("DynamicRangeCompression_delta_delta", model_dir, 
                                                scores_dirs, tasks={"audio_classification":"Audio classification"},
                                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                                from_epoch=0, to_epoch=49, epochs_skip=0, 
                                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                                title_prefix = "Paper model with Dynamic Range Compression and delta-delta features",
                                                colors = colors,
                                                plot_dir = plot_dir
                                                )
                
                plot_scores_from_multiple_dirs("PitchShift_PS1_delta_delta", model_dir, 
                                                scores_dirs, tasks={"audio_classification":"Audio classification"},
                                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                                from_epoch=0, to_epoch=49, epochs_skip=0, 
                                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                                title_prefix = "Paper model with Pitch Shift (shorter range) and delta-delta features",
                                                colors = colors,
                                                plot_dir = plot_dir
                                                )
                
                plot_scores_from_multiple_dirs("PitchShift_PS2_delta_delta", model_dir, 
                                                scores_dirs, tasks={"audio_classification":"Audio classification"},
                                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                                from_epoch=0, to_epoch=49, epochs_skip=0, 
                                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                                title_prefix = "Paper model with Pitch Shift (wider range) and delta-delta features",
                                                colors = colors,
                                                plot_dir = plot_dir
                                                )
                
                plot_scores_from_multiple_dirs("TimeStretch_delta_delta", model_dir, 
                                                scores_dirs, tasks={"audio_classification":"Audio classification"},
                                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                                from_epoch=0, to_epoch=49, epochs_skip=0, 
                                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                                title_prefix = "Paper model with Time Stretch and delta-delta features",
                                                colors = colors,
                                                plot_dir = plot_dir
                                                )
                                                
                plot_scores_from_multiple_dirs("Base_IMAGE_SHIFT_custom", model_dir, 
                                                scores_dirs, tasks={"audio_classification":"Audio classification"},
                                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                                from_epoch=0, to_epoch=49, epochs_skip=0, 
                                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                                title_prefix = "Custom model - Random spectrogram image right shift augmentation",
                                                colors = colors,
                                                plot_dir = plot_dir
                                                )

                plot_scores_from_multiple_dirs("Base_IMAGE_NOISE_custom", model_dir, 
                                                scores_dirs, tasks={"audio_classification":"Audio classification"},
                                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                                from_epoch=0, to_epoch=49, epochs_skip=0, 
                                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                                title_prefix = "Custom model - Random spectrogram image noise augmentation",
                                                colors = colors,
                                                plot_dir = plot_dir
                                                )
                                                
                plot_scores_from_multiple_dirs("Base_IMAGE_SHIFT_NOISE_custom", model_dir, 
                                                scores_dirs, tasks={"audio_classification":"Audio classification"},
                                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                                from_epoch=0, to_epoch=49, epochs_skip=0, 
                                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                                title_prefix = "Custom model - Random spectrogram image right shift and noise augmentation",
                                                colors = colors,
                                                plot_dir = plot_dir
                                                )
                                                         
                plot_scores_from_multiple_dirs("Base_IMAGE_SHIFT", model_dir, 
                                                scores_dirs, tasks={"audio_classification":"Audio classification"},
                                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                                from_epoch=0, to_epoch=49, epochs_skip=0, 
                                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                                title_prefix = "Paper model - Random spectrogram image right shift augmentation",
                                                colors = colors,
                                                plot_dir = plot_dir
                                                )

                plot_scores_from_multiple_dirs("Base_IMAGE_NOISE", model_dir, 
                                                scores_dirs, tasks={"audio_classification":"Audio classification"},
                                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                                from_epoch=0, to_epoch=49, epochs_skip=0, 
                                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                                title_prefix = "Paper model - Random spectrogram image noise augmentation",
                                                colors = colors,
                                                plot_dir = plot_dir
                                                )
                                                
                plot_scores_from_multiple_dirs("Base_IMAGE_SHIFT_NOISE", model_dir, 
                                                scores_dirs, tasks={"audio_classification":"Audio classification"},
                                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                                from_epoch=0, to_epoch=49, epochs_skip=0, 
                                                xticks_step=3, combine_tasks=False, increase_epoch_labels_by_one=True, 
                                                title_prefix = "Paper model - Random spectrogram image right shift and noise augmentation",
                                                colors = colors,
                                                plot_dir = plot_dir
                                                )
                
        ####################
        # CONFUSION MATRIX #
        ####################
        if CONFUSION_MATRIX:
                    
                plot_confusion_matrix("Base", model_dir, 
                                        tasks={"audio_classification":"Audio classification"},
                                         
                                        title_prefix = "Confusion Matrix Base",
                                        scores_on_train=False,
                                        plot_dir = plot_dir
                                        )
                
                plot_confusion_matrix("Base_custom", model_dir, 
                                        tasks={"audio_classification":"Audio classification"},
                                         
                                        title_prefix = "Confusion Matrix Base_custom",
                                        scores_on_train=False,
                                        plot_dir = plot_dir
                                        )
                                        
                plot_confusion_matrix("Base_delta", model_dir, 
                                        tasks={"audio_classification":"Audio classification"},
                                         
                                        title_prefix = "Confusion Matrix Base using Delta",
                                        scores_on_train=False,
                                        plot_dir = plot_dir
                                        )

                plot_confusion_matrix("Base_delta_delta", model_dir, 
                                        tasks={"audio_classification":"Audio classification"},
                                         
                                        title_prefix = "Confusion Matrix Base using Delta Delta",
                                        scores_on_train=False,
                                        plot_dir = plot_dir
                                        )
                
                plot_confusion_matrix("Base", model_dir, 
                                        tasks={"audio_classification":"Audio classification"},
                                         
                                        title_prefix = "Base",
                                        scores_on_train=False,
                                        plot_dir = plot_dir
                                        )

                plot_confusion_matrix("Base_IMAGE_NOISE", model_dir, 
                                        tasks={"audio_classification":"Audio classification"},
                                         
                                        title_prefix = "Paper Model - Confusion Matrix Image Noise",
                                        scores_on_train=False,
                                        plot_dir = plot_dir
                                        )
                
                plot_confusion_matrix("Base_IMAGE_NOISE_custom", model_dir, 
                                        tasks={"audio_classification":"Audio classification"},
                                         
                                        title_prefix = "Custom Model - Confusion Matrix Image Noise",
                                        scores_on_train=False,
                                        plot_dir = plot_dir
                                        )
                
                plot_confusion_matrix("Base_IMAGE_SHIFT", model_dir, 
                                        tasks={"audio_classification":"Audio classification"},
                                         
                                        title_prefix = "Paper Model - Confusion Matrix Image Shift",
                                        scores_on_train=False,
                                        plot_dir = plot_dir
                                        )
                
                plot_confusion_matrix("Base_IMAGE_SHIFT_custom", model_dir, 
                                        tasks={"audio_classification":"Audio classification"},
                                         
                                        title_prefix = "Custom Model - Confusion Matrix Image Shift",
                                        scores_on_train=False,
                                        plot_dir = plot_dir
                                        )
                
                plot_confusion_matrix("Base_IMAGE_SHIFT_NOISE", model_dir, 
                                        tasks={"audio_classification":"Audio classification"},
                                         
                                        title_prefix = "Paper Model - Confusion Matrix Image Shift",
                                        scores_on_train=False,
                                        plot_dir = plot_dir
                                        )
                
                plot_confusion_matrix("Base_IMAGE_SHIFT_NOISE_custom", model_dir, 
                                        tasks={"audio_classification":"Audio classification"},
                                         
                                        title_prefix = "Custom Model - Confusion Matrix Image Shift",
                                        scores_on_train=False,
                                        plot_dir = plot_dir
                                        )
                

        #####################
        # Comparative plots #
        #####################
        if COMPARATIVE_PLOTS:
                
                #Confronto tra tutte le audio augmentation sul modello paper
                model_names = {
                                "Base" : "Paper model - No augmentation", 
                                "PitchShift_PS1" : "Paper model - Pitch Shift (shorter range)",
                                "PitchShift_PS2" : "Paper model - Pitch Shift (wider range)",
                                "TimeStretch" : "Paper model - Time Stretch",
                                "DynamicRangeCompression" : "Paper model - Dynamic Range Compression",
                                "BackgroundNoise" : "Paper model - Background Noise"
                              }
                comparative_plots(model_names, model_dir,
                                tasks={"audio_classification":"Audio classification"},
                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                from_epoch=0, to_epoch=39, epochs_skip=0, 
                                xticks_step=3, increase_epoch_labels_by_one=True,
                                title_prefix = "Paper model audio augmentations",
                                plot_dir = plot_dir
                                ) 

                #Confronto tra tutte le audio augmentation sul modello custom
                model_names = {
                                "Base_custom" : "Custom model - No augmentation", 
                                "PitchShift_PS1_custom" : "Custom model - Pitch Shift (shorter range)",
                                "PitchShift_PS2_custom" : "Custom model - Pitch Shift (wider range)",
                                "TimeStretch_custom" : "Custom model - Time Stretch",
                                "DynamicRangeCompression_custom" : "Custom model - Dynamic Range Compression",
                                "BackgroundNoise_custom" : "Custom model - Background Noise"
                              }
                comparative_plots(model_names, model_dir,
                                tasks={"audio_classification":"Audio classification"},
                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                from_epoch=0, to_epoch=39, epochs_skip=0, 
                                xticks_step=3, increase_epoch_labels_by_one=True,
                                title_prefix = "Custom model audio augmentations",
                                plot_dir = plot_dir
                                ) 
                
                #Confronto tra Base, Base + Delta e Base + Delta Delta sul modello paper
                model_names = {
                                "Base" : "Base", 
                                "Base_delta" : "Base + Spectrogram Delta",
                                "Base_delta_delta" : "Base + Spectrogram Delta-Delta"
                              }
                comparative_plots(model_names, model_dir,
                                tasks={"audio_classification":"Audio classification"},
                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                from_epoch=0, to_epoch=39, epochs_skip=0, 
                                xticks_step=3, increase_epoch_labels_by_one=True,
                                title_prefix = "Paper model with delta and delta-delta",
                                plot_dir = plot_dir
                                ) 

                #Confronto tra tutte le audio augmentation DELTA sul modello paper
                model_names = {
                                "Base_delta" : "Paper model with delta - No augmentation", 
                                "PitchShift_PS1_delta" : "Paper model with delta - Pitch Shift (shorter range)",
                                "PitchShift_PS2_delta" : "Paper model with delta - Pitch Shift (wider range)",
                                "TimeStretch_delta" : "Paper model with delta - Time Stretch",
                                "DynamicRangeCompression_delta" : "Paper model with delta - Dynamic Range Compression",
                                "BackgroundNoise_delta" : "Paper model with delta - Background Noise"
                              }
                comparative_plots(model_names, model_dir,
                                tasks={"audio_classification":"Audio classification"},
                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                from_epoch=0, to_epoch=39, epochs_skip=0, 
                                xticks_step=3, increase_epoch_labels_by_one=True,
                                title_prefix = "Paper model audio augmentations with delta",
                                plot_dir = plot_dir
                                ) 
                
                #Confronto tra tutte le audio augmentation DELTA DELTA sul modello paper
                model_names = {
                                "Base_delta_delta" : "Paper model with delta-delta - No augmentation", 
                                "PitchShift_PS1_delta_delta" : "Paper model with delta-delta - Pitch Shift (shorter range)",
                                "PitchShift_PS2_delta_delta" : "Paper model with delta-delta - Pitch Shift (wider range)",
                                "TimeStretch_delta_delta" : "Paper model with delta-delta - Time Stretch",
                                "DynamicRangeCompression_delta_delta" : "Paper model with delta-delta - Dynamic Range Compression",
                                "BackgroundNoise_delta_delta" : "Paper model with delta-delta - Background Noise"
                              }
                comparative_plots(model_names, model_dir,
                                tasks={"audio_classification":"Audio classification"},
                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                from_epoch=0, to_epoch=39, epochs_skip=0, 
                                xticks_step=3, increase_epoch_labels_by_one=True,
                                title_prefix = "Paper model audio augmentations with delta-delta",
                                plot_dir = plot_dir
                                ) 

                #Confronto tra Base, Random Image Shift, Random Image Noise, Random Image Shift + Random Image Noise sul modello custom
                
                model_names = {
                                "Base" : "Paper model - No augmentation", 
                                "Base_IMAGE_SHIFT" : "Paper model with spectrogram Image Shift",
                                "Base_IMAGE_NOISE" : "Paper model with spectrogram Image Noise",
                                "Base_IMAGE_SHIFT_NOISE" : "Paper model with spectrogram Image Shift and Noise",
                              }
                comparative_plots(model_names, model_dir,
                                tasks={"audio_classification":"Audio classification"},
                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                from_epoch=0, to_epoch=39, epochs_skip=0, 
                                xticks_step=3, increase_epoch_labels_by_one=True,
                                title_prefix = "Paper model spectrogram image augmentations",
                                plot_dir = plot_dir
                                )
                
                #Confronto tra Base, Random Image Shift, Random Image Noise, Random Image Shift + Random Image Noise sul modello paper
                model_names = {
                                "Base_custom" : "Custom model - No augmentation", 
                                "Base_IMAGE_SHIFT_custom" : "Custom model with spectrogram Image Shift",
                                "Base_IMAGE_NOISE_custom" : "Custom model with spectrogram Image Noise",
                                "Base_IMAGE_SHIFT_NOISE_custom" : "Custom model with spectrogram Image Shift and Noise",
                              }
                comparative_plots(model_names, model_dir,
                                tasks={"audio_classification":"Audio classification"},
                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                from_epoch=0, to_epoch=39, epochs_skip=0, 
                                xticks_step=3, increase_epoch_labels_by_one=True,
                                title_prefix = "Custom model spectrogram image augmentations",
                                plot_dir = plot_dir
                                )
                
                #Confronto tra Base, Migliore tra i delta, Migliore tra le augmentation audio, Migliore tra le augmentation immagine
                
                model_names = {
                                "Base" : "Paper model - No augmentation", 
                                "DynamicRangeCompression" : "Best of the audio augmentations (Dynamic Range Compression)", 
                                "Base_IMAGE_SHIFT_NOISE" : "Best of the image augmentations (SHIFT + NOISE)",
                              }
                comparative_plots(model_names, model_dir,
                                tasks={"audio_classification":"Audio classification"},
                                metrics={"F1-macro":["f1"], "Accuracy":["accuracy"]},
                                from_epoch=0, to_epoch=39, epochs_skip=0, 
                                xticks_step=3, increase_epoch_labels_by_one=True,
                                title_prefix = "Paper model comparison of the best augmentations",
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

                #TODO: Scegliere un altro modello in cui il training è stato particolarmente BUONO (tipo le image augmentation)
                create_gradient_flow_gif("Base_IMAGE_SHIFT_NOISE", model_dir, plot_dir, cropX=cropX, cropY=cropY)

        ################
        # BEST  SCORES #
        ################
        if BEST_SCORES:
                #TODO: Semplicemente aggiungere a questa lista tutti i modelli
                model_names = {
                        "Base",
                        "PitchShift_PS1",
                        "PitchShift_PS2",
                        "TimeStretch",
                        "DynamicRangeCompression",
                        "BackgroundNoise",
                        
                        "Base_custom",
                        "PitchShift_PS1_custom",
                        "PitchShift_PS2_custom",
                        "TimeStretch_custom",
                        "DynamicRangeCompression_custom",
                        "BackgroundNoise_custom",

                        "Base_IMAGE_SHIFT",
                        "Base_IMAGE_NOISE",
                        "Base_IMAGE_SHIFT_NOISE",
                        
                        "Base_IMAGE_SHIFT_custom",
                        "Base_IMAGE_NOISE_custom",
                        "Base_IMAGE_SHIFT_NOISE_custom",
                        
                        "Base_delta",
                        "PitchShift_PS1_delta",
                        "PitchShift_PS2_delta",
                        "TimeStretch_delta",
                        "DynamicRangeCompression_delta",
                        "BackgroundNoise_delta",
                        
                        "Base_delta_delta", 
                        "PitchShift_PS1_delta_delta",
                        "PitchShift_PS2_delta_delta",
                        "TimeStretch_delta_delta",
                        "DynamicRangeCompression_delta_delta",
                        "BackgroundNoise_delta_delta"
                        }

                with open(os.path.join(model_dir,"best_scores.txt"), "w") as f:
                        for model_name in model_names:
                                _, best_scores_str = get_best_epoch_scores(model_name, model_dir,
                                        metrics = {
                                                        "accuracy": "Accuracy",
                                                        "precision": "Precision",
                                                        "recall": "Recall",
                                                        "f1": "F1 Macro Avg.",
                                                        "distribution": "Distribution", 
                                                        "confusion matrix": "Confusion Matrix"
                                                  }
                                                )
                                f.write(best_scores_str)
        #################
        # DELTA  SCORES #
        #################
        if PREPROCESSING_PERFORMANCE_DELTA_COMPARISONS:

                                
                classes_names = ["air_conditioner","car_horn","children_playing","dog_bark","drilling","engine_idling","gun_shot","jackhammer","siren","street_music", "All classes"]

                #TODO: Generare delta scores delle classi su i seguenti insiemi di modelli:
                # 1) Base paper con tutte augmentation
                dict_augmentation_to_test = {
                                            "Base" : "Base",
                                            "PitchShift_PS1" : "PS1",
                                            "PitchShift_PS2" : "PS2",
                                            "BackgroundNoise" : "BG",
                                            "DynamicRangeCompression" : "DRC",
                                            "TimeStretch" : "TS"
                                            }
                plot_delta_on_metric(model_dir, "all", dict_augmentation_to_test,
                                    "Base", ["Δ Classification Accuracies", "class"],
                                    "value", "class", classes_names = classes_names, horizontal=True, plot_dir="plots")
                # 2) Base custom con tutte augmentation
                dict_augmentation_to_test = {
                                            "Base_custom" : "Base",
                                            "PitchShift_PS1_custom" : "PS1",
                                            "PitchShift_PS2_custom" : "PS2",
                                            "BackgroundNoise_custom" : "BG",
                                            "DynamicRangeCompression_custom" : "DRC",
                                            "TimeStretch_custom" : "TS"
                                            }
                plot_delta_on_metric(model_dir, "all", dict_augmentation_to_test,
                                    "Base_custom", ["Δ Classification Accuracies", "class"],
                                    "value", "class", classes_names = classes_names, horizontal=True, plot_dir="plots") 
                # 3) Base paper delta con tutte augmentation 
                dict_augmentation_to_test = {
                                            "Base_delta" : "Base",
                                            "PitchShift_PS1_delta" : "PS1",
                                            "PitchShift_PS2_delta" : "PS2",
                                            "BackgroundNoise_delta" : "BG",
                                            "DynamicRangeCompression_delta" : "DRC",
                                            "TimeStretch_delta" : "TS"
                                            }
                plot_delta_on_metric(model_dir, "all", dict_augmentation_to_test,
                                    "Base_delta", ["Δ Classification Accuracies", "class"],
                                    "value", "class", classes_names = classes_names, horizontal=True, plot_dir="plots")
                # 4) Base paper delta delta tutte augmentation
                #plot delta on total accuracies
                dict_augmentation_to_test = {
                                            "Base_delta_delta" : "Base",
                                            "PitchShift_PS1_delta_delta" : "PS1",
                                            "PitchShift_PS2_delta_delta" : "PS2",
                                            "BackgroundNoise_delta_delta" : "BG",
                                            "DynamicRangeCompression_delta_delta" : "DRC",
                                            "TimeStretch_delta_delta" : "TS"
                                            }
                plot_delta_on_metric(model_dir, "all", dict_augmentation_to_test,
                                    "Base_delta_delta", ["Δ Classification Accuracies", "class"],
                                    "value", "class", classes_names = classes_names, horizontal=True, plot_dir="plots")
                
                # 5) Paper con tutte le image augmentation 
                dict_augmentation_to_test = {
                                            "Base" : "Paper model - Base",
                                            "Base_IMAGE_SHIFT" : "Paper model - Spectrogram right shift",
                                            "Base_IMAGE_NOISE" : "Paper model - Spectrogram random noise",
                                            "Base_IMAGE_SHIFT_NOISE" : "Paper model - Spectrogram random noise + Spectrogram right shift"
                                            }
                plot_delta_on_metric(model_dir, "all", dict_augmentation_to_test, 
                                    "Base", ["Δ Classification Accuracy", "class"],
                                    "value", "class", classes_names = classes_names, horizontal=False, plot_dir="plots")
                                    
                # 5) Custom con tutte le image augmentation 
                dict_augmentation_to_test = {
                                            "Base_custom" : "Custom model - Base",
                                            "Base_IMAGE_SHIFT_custom" : "Custom model - Spectrogram right shift",
                                            "Base_IMAGE_NOISE_custom" : "Custom model - Spectrogram random noise",
                                            "Base_IMAGE_SHIFT_NOISE_custom" : "Custom model - Spectrogram random noise + Spectrogram right shift"
                                            }
                plot_delta_on_metric(model_dir, "all", dict_augmentation_to_test, 
                                    "Base_custom", ["Δ Classification Accuracy", "class"],
                                    "value", "class", classes_names = classes_names, horizontal=False, plot_dir="plots")
                                    
                # 7) Paper con tutte le migliori 
                dict_augmentation_to_test = {
                                            "Base" : "Paper model - Base",
                                            "Base_IMAGE_SHIFT_NOISE" : "Paper model - Spectrogram random noise + Spectrogram right shift",
                                            "DynamicRangeCompression" : "Paper model - DRC",
                                            }
                plot_delta_on_metric(model_dir, "all", dict_augmentation_to_test, 
                                    "Base", ["Δ Classification Accuracy", "class"],
                                    "value", "class", classes_names = classes_names, horizontal=False, plot_dir="plots")
                
                
                #TODO: Generare delta scores dell'intero modello per i seguenti insiemi di modelli:
                # 1) Base paper con tutte augmentation
                dict_augmentation_to_test = {
                                            "Base" : "Base",
                                            "PitchShift_PS1" : "PS1",
                                            "PitchShift_PS2" : "PS2",
                                            "BackgroundNoise" : "BG",
                                            "DynamicRangeCompression" : "DRC",
                                            "TimeStretch" : "TS"
                                            }
                plot_delta_on_metric(model_dir, "accuracy", dict_augmentation_to_test, 
                                    "Base", ["Δ Classification Accuracy", "class"],
                                    "value", "class", horizontal=False, plot_dir="plots")

                plot_delta_on_metric(model_dir, "f1",dict_augmentation_to_test,
                                    "Base", ["Δ f1-score", "augmentations"], 
                                    "f1_score" , "augmentations",horizontal=False,plot_dir="plots")

                # 2) Base custom con tutte augmentation
                dict_augmentation_to_test = {
                                            "Base_custom" : "Base",
                                            "PitchShift_PS1_custom" : "PS1",
                                            "PitchShift_PS2_custom" : "PS2",
                                            "BackgroundNoise_custom" : "BG",
                                            "DynamicRangeCompression_custom" : "DRC",
                                            "TimeStretch_custom" : "TS"
                                            }
                plot_delta_on_metric(model_dir, "accuracy", dict_augmentation_to_test, 
                                    "Base_custom", ["Δ Classification Accuracy", "class"],
                                    "value", "class", horizontal=False, plot_dir="plots")
                
                plot_delta_on_metric(model_dir, "f1",dict_augmentation_to_test,
                                    "Base_custom", ["Δ f1-score", "augmentations"], 
                                    "f1_score" , "augmentations",horizontal=False,plot_dir="plots")

                # 3) Base paper con tutte augmentation + delta
                dict_augmentation_to_test = {
                                            "Base_delta" : "Base",
                                            "PitchShift_PS1_delta" : "PS1",
                                            "PitchShift_PS2_delta" : "PS2",
                                            "BackgroundNoise_delta" : "BG",
                                            "DynamicRangeCompression_delta" : "DRC",
                                            "TimeStretch_delta" : "TS"
                                            }
                plot_delta_on_metric(model_dir, "accuracy", dict_augmentation_to_test, 
                                    "Base_delta", ["Δ Classification Accuracy", "class"],
                                    "value", "class", horizontal=False, plot_dir="plots")
                                    
                plot_delta_on_metric(model_dir, "f1",dict_augmentation_to_test,
                                    "Base_delta", ["Δ f1-score", "augmentations"], 
                                    "f1_score" , "augmentations",horizontal=False,plot_dir="plots")
                # 4) Base paper con tutte augmentation + delta-delta 
                
                #plot delta on total accuracy
                dict_augmentation_to_test = {
                                            "Base_delta_delta" : "Base",
                                            "PitchShift_PS1_delta_delta" : "PS1",
                                            "PitchShift_PS2_delta_delta" : "PS2",
                                            "BackgroundNoise_delta_delta" : "BG",
                                            "DynamicRangeCompression_delta_delta" : "DRC",
                                            "TimeStretch_delta_delta" : "TS"
                                            }
                plot_delta_on_metric(model_dir, "accuracy", dict_augmentation_to_test, 
                                    "Base_delta_delta", ["Δ Classification Accuracy", "class"],
                                    "value", "class", horizontal=False, plot_dir="plots")
                
                plot_delta_on_metric(model_dir, "f1",dict_augmentation_to_test,
                                    "Base_delta_delta", ["Δ f1-score", "augmentations"], 
                                    "f1_score" , "augmentations",horizontal=False,plot_dir="plots")
                # 5) Paper con tutte le image augmentation 
                

        ###################################
        # PLOT TRAIN/TEST ACCURACY DELTAS #
        ###################################
        if PLOT_TRAIN_TEST_ACCURACY_DELTAS:
                
                # 1) Base paper con tutte augmentation
                model_names = {
                                "Base" : "Paper model - No augmentation", 
                                "PitchShift_PS1" : "Paper model - Pitch Shift (shorter range)",
                                "PitchShift_PS2" : "Paper model - Pitch Shift (wider range)",
                                "TimeStretch" : "Paper model - Time Stretch",
                                "DynamicRangeCompression" : "Paper model - Dynamic Range Compression",
                                "BackgroundNoise" : "Paper model - Background Noise"
                              }
                plot_train_test_accuracy_delta(model_dir, model_names, 
                                                metrics = {"accuracy" : "Accuracy"},
                                                tasks = {"audio_classification" : "Audio classification"},
                                                show = False
                                                )
                # 2) Base custom con tutte augmentation
                model_names = {
                                "Base_custom" : "Custom model - No augmentation", 
                                "PitchShift_PS1_custom" : "Custom model - Pitch Shift (shorter range)",
                                "PitchShift_PS2_custom" : "Custom model - Pitch Shift (wider range)",
                                "TimeStretch_custom" : "Custom model - Time Stretch",
                                "DynamicRangeCompression_custom" : "Custom model - Dynamic Range Compression",
                                "BackgroundNoise_custom" : "Custom model - Background Noise"
                              }
                plot_train_test_accuracy_delta(model_dir, model_names, 
                                                metrics = {"accuracy" : "Accuracy"},
                                                tasks = {"audio_classification" : "Audio classification"},
                                                show = False
                                                )
                # 3) Base paper con tutte augmentation + delta
                model_names = {
                                "Base_delta" : "Paper model with delta - No augmentation", 
                                "PitchShift_PS1_delta" : "Paper model with delta - Pitch Shift (shorter range)",
                                "PitchShift_PS2_delta" : "Paper model with delta - Pitch Shift (wider range)",
                                "TimeStretch_delta" : "Paper model with delta - Time Stretch",
                                "DynamicRangeCompression_delta" : "Paper model with delta - Dynamic Range Compression",
                                "BackgroundNoise_delta" : "Paper model with delta - Background Noise"
                              }
                plot_train_test_accuracy_delta(model_dir, model_names, 
                                                metrics = {"accuracy" : "Accuracy"},
                                                tasks = {"audio_classification" : "Audio classification"},
                                                show = False
                                                )
                # 4) Base paper con tutte augmentation + delta-delta
                model_names = {
                                "Base_delta_delta" : "Paper model with delta-delta - No augmentation", 
                                "PitchShift_PS1_delta_delta" : "Paper model with delta-delta - Pitch Shift (shorter range)",
                                "PitchShift_PS2_delta_delta" : "Paper model with delta-delta - Pitch Shift (wider range)",
                                "TimeStretch_delta_delta" : "Paper model with delta-delta - Time Stretch",
                                "DynamicRangeCompression_delta_delta" : "Paper model with delta-delta - Dynamic Range Compression",
                                "BackgroundNoise_delta_delta" : "Paper model with delta-delta - Background Noise"
                              }
                plot_train_test_accuracy_delta(model_dir, model_names, 
                                                metrics = {"accuracy" : "Accuracy"},
                                                tasks = {"audio_classification" : "Audio classification"},
                                                show = False
                                                )
                # 5) Paper con tutte le image augmentation
                
                model_names = {
                                "Base" : "Paper model - No augmentation", 
                                "Base_IMAGE_SHIFT" : "Paper model with spectrogram Image Shift",
                                "Base_IMAGE_NOISE" : "Paper model with spectrogram Image Noise",
                                "Base_IMAGE_SHIFT_NOISE" : "Paper model with spectrogram Image Shift and Noise",
                              }
                plot_train_test_accuracy_delta(model_dir, model_names, 
                                                metrics = {"accuracy" : "Accuracy"},
                                                tasks = {"audio_classification" : "Audio classification"},
                                                show = False
                                                )
                
                # 6) Custom con tutte le image augmentation
                model_names = {
                                "Base_custom" : "Custom model - No augmentation", 
                                "Base_IMAGE_SHIFT_custom" : "Custom model with spectrogram Image Shift",
                                "Base_IMAGE_NOISE_custom" : "Custom model with spectrogram Image Noise",
                                "Base_IMAGE_SHIFT_NOISE_custom" : "Custom model with spectrogram Image Shift and Noise",
                              }
                plot_train_test_accuracy_delta(model_dir, model_names, 
                                                metrics = {"accuracy" : "Accuracy"},
                                                tasks = {"audio_classification" : "Audio classification"},
                                                show = False
                                                )
                # 7) Base paper + delta + delta-delta
                model_names = {
                                "Base" : "Base", 
                                "Base_delta" : "Base + Spectrogram Delta",
                                "Base_delta_delta" : "Base + Spectrogram Delta-Delta"
                              }
                plot_train_test_accuracy_delta(model_dir, model_names, 
                                                metrics = {"accuracy" : "Accuracy"},
                                                tasks = {"audio_classification" : "Audio classification"},
                                                show = False
                                                )
                # 8) Delta modelli migliori
                
                model_names = {
                                "Base" : "Paper model - No augmentation", 
                                "DynamicRangeCompression" : "Best of the audio augmentations (Dynamic Range Compression)", 
                                "Base_IMAGE_SHIFT_NOISE_custom" : "Best of the image augmentations (SHIFT + NOISE)",
                              }
                plot_train_test_accuracy_delta(model_dir, model_names, 
                                                metrics = {"accuracy" : "Accuracy"},
                                                tasks = {"audio_classification" : "Audio classification"},
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

                #print(samples_one_for_each_class)
                
                #Preprocess the collected samples
                preprocessors = [
                                PitchShift(values = [-3.5]),
                                MUDADynamicRangeCompression(),
                                BackgroundNoise({
                                "street_scene_1" : "150993__saphe__street-scene-1.wav",
                                #    "street_scene_3" : "173955__saphe__street-scene-3.wav",
                                #    "street_valencia" : "207208__jormarp__high-street-of-gandia-valencia-spain.wav",
                                #    "city_park_tel_aviv" : "268903__yonts__city-park-tel-aviv-israel.wav",
                                }, files_dir = os.path.join(dataset_dir, "UrbanSound8K-JAMS", "background_noise")),
                                TimeStretch(values = [0.5])
                                ]

                for class_name, sample in samples_one_for_each_class.items():
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

#TODO: Popolare il dizionario models con i modelli di cui vogliamo generare le saliency map
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
                