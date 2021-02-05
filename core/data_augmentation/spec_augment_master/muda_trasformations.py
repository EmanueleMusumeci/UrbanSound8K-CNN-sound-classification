
"""
# Loading audio from disk with an existing jams
j_orig = jams.load('existing_jams_file.jams')
j_orig = muda.load_jam_audio(existing_jams, 'orig.ogg')
# Ready to go!

# Loading in-memory audio (y, sr) with an existing jams
j_orig = jams.load('existing_jams_file.jams')
j_orig = muda.jam_pack(existing_jams, _audio=dict(y=y, sr=sr))
# Ready to go!
"""
import os 
import jams
import muda

if __name__ == "__main__":
    
    EXAMPLE_NOTEBOOK = True

    base_dir = os.path.dirname(os.path.realpath(__file__))
    base_dir = base_dir.replace("data_augmentation","")
    print("--------------------: ",base_dir)
    base_dir = base_dir.replace("\\spec_augment_master","")
    print(base_dir)
    DATASET_DIR = os.path.join(base_dir,"data")
    DATASET_NAME = "UrbanSound8K"
    UrbanSound8K = os.path.join(DATASET_DIR,DATASET_NAME)
    audio = os.path.join(UrbanSound8K,"audio")
    fold1 = os.path.join(audio,"fold1")
    sample = os.path.join(fold1,"7061-6-0-0.wav")
    print(sample)
    muda_augmentation = os.path.join(DATASET_DIR,"muda_augmentation")
    jam = os.path.join(muda_augmentation,"7061-6-0-0.jams")
    print(jam)

    # Loading audio from disk with an existing jams

    j_orig = jams.load(jam)
    j_orig = muda.load_jam_audio(j_orig, sample)
    #print(j_orig)
    print(muda.__version__)

    pitch = muda.deformers.LinearPitchShift(n_samples=5, lower=-1, upper=1)

    #background = muda.deformers.
    
    for i, jam_out in enumerate(pitch.transform(j_orig)):
        muda.save('output_{:02d}.ogg'.format(i),'output_{:02d}.jams'.format(i),jam_out)
    