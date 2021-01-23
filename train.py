import os

import torch
try:
    from Dataset import SoundDatasetFold
    from DataLoader import DataLoader
    from nn.convolutional_model import ConvolutionalNetwork
    from nn.feed_forward_model import FeedForwardNetwork
    from data_augmentation.image_transformations import *
    from data_augmentation.audio_transformations import *
    from Trainer import *
except:
    pass

INSTANCE_NAME = "PROVA"
BATCH_SIZE = 128
USE_CNN = True
APPLY_IMAGE_AUGMENTATIONS = True
APPLY_AUDIO_AUGMENTATIONS = True

DEBUG_TIMING = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
  from google.colab import drive
  base_dir = "/content/drive/My Drive/Neural Networks Project"
  DATASET_DIR = "/content/drive/data"
except:
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

#Image augmentations
if APPLY_IMAGE_AUGMENTATIONS:
    right_shift_transformation = SpectrogramShift(input_size=CNN_INPUT_SIZE,width_shift_range=4,shift_prob=0.9)
    left_shift_transformation = SpectrogramShift(input_size=CNN_INPUT_SIZE,width_shift_range=4,shift_prob=0.9, left=True)
    random_side_shift_transformation = SpectrogramShift(input_size=CNN_INPUT_SIZE,width_shift_range=4,shift_prob=0.9, random_side=True)
    background_noise_transformation = SpectrogramAddGaussNoise(input_size=CNN_INPUT_SIZE,prob_to_have_noise=0.55)
else:
    right_shift_transformation = None
    left_shift_transformation = None
    random_side_shift_transformation = None
    background_noise_transformation = None

#Audio augmentations
if APPLY_AUDIO_AUGMENTATIONS:
    random_pitch_shift = PitchShift([-3.5, -2.5, 2.5, 3.5], debug_time=DEBUG_TIMING)
    random_time_stretch = TimeStretch([0.81, 0.93, 1.07, 1.23], debug_time=DEBUG_TIMING)
else:
    random_pitch_shift = None
    random_time_stretch = None

train_dataset = SoundDatasetFold(DATASET_DIR, DATASET_NAME, 
                            folds = [1], 
                            shuffle = True, 
                            generate_spectrograms = USE_CNN, 
                            shift_transformation = right_shift_transformation, 
                            background_noise_transformation = background_noise_transformation, 
                            time_stretch_transformation = random_time_stretch,
                            pitch_shift_transformation = random_pitch_shift, 
                            spectrogram_frames_per_segment = spectrogram_frames_per_segment, 
                            spectrogram_bands = spectrogram_bands, 
                            compute_deltas=True, 
                            compute_delta_deltas=True, 
                            test = False, 
                            progress_bar = True,
                            selected_classes=selected_classes,
                            select_percentage_of_dataset=DATASET_PERCENTAGE,
                            debug_preprocessing_time=DEBUG_TIMING
                            )   

test_dataset = SoundDatasetFold(DATASET_DIR, DATASET_NAME, 
                            folds = [2], 
                            shuffle = False, 
                            generate_spectrograms = USE_CNN, 
                            spectrogram_frames_per_segment = spectrogram_frames_per_segment, 
                            spectrogram_bands = spectrogram_bands, 
                            compute_deltas=True, 
                            compute_delta_deltas=True, 
                            test = True, 
                            progress_bar = True,
                            selected_classes=selected_classes,
                            select_percentage_of_dataset=DATASET_PERCENTAGE,
                            debug_preprocessing_time=DEBUG_TIMING
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

loss_function = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters())
#optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
#lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

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
