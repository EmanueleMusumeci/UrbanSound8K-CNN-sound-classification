import sox
import os
import librosa
# create transformer
tfm = sox.Transformer()
# trim the audio between 5 and 10.5 seconds.
tfm.trim(5, 10.5)
# apply compression
tfm.compand()
# apply a fade in and fade out
tfm.fade(fade_in_len=1.0, fade_out_len=0.5)
# create an output file.
base_dir = os.path.dirname(os.path.realpath(__file__))
DATASET_DIR = os.path.join(base_dir,"data")
print(librosa.__version__)

name_file = "7061-6-0-0"
type_file = ".wav"

sound_file = os.path.join(DATASET_DIR,"UrbanSound8K","audio","fold1",name_file+type_file)
sound_file = sound_file.replace("data_augmentation\\","")

tfm.build_file(sound_file, 'audio.aiff')
# or equivalently using the legacy API
tfm.build(sound_file, 'audio.aiff')
# get the output in-memory as a numpy array
# by default the sample rate will be the same as the input file
array_out = tfm.build_array(input_filepath=sound_file)
# see the applied effects
print(tfm.effects_log)