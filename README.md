This repo contains an implementation of a classifier based on a convolutional neural network, to solve the [environmental sound classification](https://paperswithcode.com/task/environmental-sound-classification) task. The CNN is trained using audio data taken from the [UrbanSound8K dataset](https://urbansounddataset.weebly.com/urbansound8k.html), with the addition of various feature extraction techniques:
1. Audio augmentations, as seen in the paper [Deep Convolutional Neural Networks and Data Augmentation for Environmental Sound Classification, Justin Salamon and Juan Pablo Bello](https://arxiv.org/pdf/1608.04363v2.pdf). In particular, the _Dynamic Range Compression_ augmentation is performed using the [MUDA library](https://github.com/bmcfee/muda).
2. Spectrogram deltas and delta-deltas, as seen in the paper [Environmental Sound Classification with Convolutional Neural Networks, Karol J. Piczak](https://www.karolpiczak.com/papers/Piczak2015-ESC-ConvNet.pdf)
3. Spectrogram image augmentation techniques (mostly found on the web and by experimentation)

# Setup
1. As a preliminary step it is strongly advised to setup a conda virtual environment with _Python 3.7_ (which can be done by following the guide [here](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html#managing-environments)).
2. Then of course clone this project using:
```
git clone https://github.com/EmanueleMusumeci/UrbanSound8K-CNN-sound-classification
```
3. Make sure that the  CUDA toolkit was correctly installed.
4. Install _pytorch_, _torchvision_ and _torchaudio_ using conda:
```
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```
5. The root folder of this project contains _requirements.txt_, a file that lists all the packages required to run this project. Install all the required packages with:
```
pip install -r requirements.txt
```
## Getting the MUDA library to work on Windows
While there shouldn't be any problem with installing the _MUDA library_ on Ubuntu, there might be problems importing the _sox_ library on Windows (which happens inside the MUDA library). Please follow [this guide](https://github.com/JoFrhwld/FAVE/wiki/Sox-on-Windows) to fix this problem on Windows.

# Download the _UrbanSound8K_ dataset
1. Follow the instructions at the [UrbanSound8K dataset webpage](https://urbansounddataset.weebly.com/urbansound8k.html) to download the dataset
2. Create a folder named "_data_" in the root directory of the project (with write permissions as well if on Ubuntu)
2. Extract the downloaded dataset _UrbanSound8K.tar.gz_ inside the folder just created

NOTICE: the dataset is divided into folds, each comprising up to 1000 samples (audio clips), totalling to 6GB of data.

# (Optional) Dataset compacting
Although this is step is optional, it is **STRONGLY ADVISED** to first perform this step as the preprocessing of this dataset is computationally heavy (just an epoch might require up to 1h), while after performing this step (which usually requires around 2h30m), an epoch requires just up to 2 minutes.

To achieve this results, the dataset is loaded fold by fold and spectrograms are generated for each clip. If preprocessing is applied, preprocessed clips and spectrograms are generated for each preprocessing value (the model will then proceed to load a preprocessed clip or spectrogram for a random preprocessing value, at training time). All this data will then be saved to a _memory-mapped_ file to speed up loading (up to a factor of 13X).

To perform this step open a terminal in the root directory of the project and run:
```
python compact_dataset.py -h
```
to see all possible command line arguments. 

NOTICE: To prevent overwriting previously compacted folds, an attempt to do so will result in the termination of the program. To bypass this failsafe use the `--overwrite_existing_folds` argument when launching the script.

If you have some issues with running python scripts from command-line or just don't like it, you can manually run the script 
`compact_dataset_manual.py
` that will compact the dataset with standard settings.

NOTICE: Also in this case, to circumvent the overwrite failsafe, you'll have to manually set the flag OVERWRITE_EXISTING_FOLDS to True.

# Start training

To start the training you'll have to run the `train.py` script as follows:
```
python train.py -h
```
to see all possible command-line arguments.

A generic use case, where no preprocessing is applied, is the following:

```
python train.py -n TRAINING_INSTANCE_NAME --dataset_dir "data" --epochs 50 
```
where TRAINING_INSTANCE_NAME is the name we want to give to this training instance. This command launches the training without using pre-compacted data, which is **STRONGLY DISCOURAGED** (training time will be way longer, see previous section for more info).

Instead, if you have previously compacted audio clips use the `--load_compacted_audio` argument, although it is advisable to compact spectrograms instead (which is done by default when running the `compact_dataset.py` or the `compact_dataset_manual.py` script) and train using the `--load_compacted_spectrograms` argument (using compacted spectrograms allows the maximum speed-up).

To just train with few samples and without any preprocessing, just to have a "taste" of the model, use the
`--test_mode` argument.

To try a deeper model instead of the one in the paper (1), use the `--custom_cnn` argument.

To apply a preprocessing, use the argument `--preprocessing_name PREPROCESSING_NAME` where _PREPROCESSING_NAME_ can be one of the following:
- PitchShift1 
- PitchShift2 
- TimeStretch 
- DynamicRangeCompression 
- BackgroundNoise

(each one of them is discussed in the _report.pdf_)

If a model with the same _TRAINING_INSTANCE_NAME_ was previously generated, the script will terminate to avoid overwriting it. To disable this failsafe use the  
`--disable_model_overwrite_protection` argument.

`--compute_deltas` and `--compute_delta_deltas` will apply the preprocessing techniques described in paper (2).

`--apply_spectrogram_image_background_noise` and `--apply_spectrogram_image_shift` will instead apply the spectrogram image augmentation techniques.

To tune regularization use `--dropout_probability DROPOUT_PROBABILITY` or `--weight_decay WEIGHT_DECAY`.

NOTICE: if you don't want to use command-line arguments, you can launch the script `train_manual.py`, where every setting has to be edited manually.
