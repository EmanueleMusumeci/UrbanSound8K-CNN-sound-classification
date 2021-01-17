import os

from Dataset import SoundDatasetFold
from DataLoader import DataLoader
from nn.convolutional_model import ConvolutionalNetwork
from nn.feed_forward_model import FeedForwardNetwork
from image_transformations import *

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
