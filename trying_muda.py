#   https://github.com/GorillaBus/urban-audio-classifier/blob/master/5-data-augmentation.ipynb
import os
import librosa
import soundfile as sf

"""
import muda
import jams

"""
base_dir = os.path.dirname(os.path.realpath(__file__))
DATASET_DIR = os.path.join(base_dir,"data")
ps_dir = os.path.join(DATASET_DIR,"ps")

DATASET_NAME = os.path.join(DATASET_DIR,"UrbanSound8K")
US8K_AUDIO_PATH = os.path.join(DATASET_NAME,"audio")

fold1 = os.path.join(US8K_AUDIO_PATH,"fold1")
"""

# Loading data from disk
j_orig = muda.load_jam_audio(jams.JAMS(),os.path.join(fold1,'7061-6-0-0.wav'))
print(j_orig)
# Ready to go!

# Loading audio from disk with an existing jams
j_orig = jams.load('existing_jams_file.jams')
j_orig = muda.load_jam_audio(existing_jams, 'orig.ogg')
# Ready to go!

# Loading in-memory audio (y, sr) with an existing jams
j_orig = jams.load('existing_jams_file.jams')
j_orig = muda.jam_pack(existing_jams, _audio=dict(y=y, sr=sr))
# Ready to go!


"""

print(librosa.__version__)
print(base_dir)
tone_steps = [-1, -2, 1, 2]
#total = len(metadata) * len(tone_steps)
print(ps_dir)

count = 0
for tone_step in tone_steps:
    # Generate new pitched audio
    #for index, row in metadata.iterrows():        
    #curr_fold = str(row['fold'])
    #curr_file_path = audio_path + '/fold' + curr_fold + '/' + row['slice_file_name']
    curr_file_path = os.path.join(fold1,'7061-6-0-0.wav')

    # Pitch Shift sub-dir inside current fold dir
    curr_ps_path = ps_dir +  '/pitch_' + str(tone_step)

    # Create sub-dir if it does not exist
    if not os.path.exists(curr_ps_path):
        os.makedirs(curr_ps_path)
    
    output_path = curr_ps_path + '/' + '7061-6-0-0.wav'
    
    # Skip when file already exists
    if (os.path.isfile(output_path)):
        count += 1 
        continue
    
    y, sr = librosa.load(curr_file_path)  
    y_changed = librosa.effects.pitch_shift(y, sr, n_steps=tone_step)
    #librosa.output.write_wav(output_path, y_changed, sr)

    sf.write(output_path,y_changed,sr)
    count += 1 
    
    #clear_output(wait=True)
    #print("Progress: {}/{}".format(count, total))
    #print("Last file: ", row['slice_file_name'])