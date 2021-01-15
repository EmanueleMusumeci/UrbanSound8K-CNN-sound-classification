import torch, torchvision
import copy

from torchvision import transforms
import numpy as np

class AudioTimeStretch(object):
    """Aggiunge rumore Gaussiano al vettore dello spettogramma"""
    #costruttore

    def __init__(self, input_size, gaussian_mean=0.0, gaussian_std=None, prob_to_have_noise=1.0):

        assert isinstance(input_size, tuple), "Input size must be a tuple (input_width, input_height)"
        self.input_size = input_size
        
        if gaussian_std is None:
            self.gaussian_std = gaussian_std
        else:
            assert isinstance(gaussian_std, float), "Gaussian std. dev must be a float"
            assert gaussian_std > 0.0,  "gaussian_std must be positive"
            self.gaussian_std = gaussian_std
        
        assert isinstance(prob_to_have_noise, float), "prob_to_have_noise must be a float"
        assert prob_to_have_noise > 0.0 and prob_to_have_noise <= 1.0, "prob_to_have_noise must be in range (0.0, 1.0)"
        self.prob_to_have_noise = prob_to_have_noise

        self.gaussian_mean = gaussian_mean 
    
    def __call__(self, img, std_factor = 0.03, noise_mask = None, debug = False):
        if np.random.random() > self.prob_to_have_noise <= 1.0:
            return img, np.zeros(self.input_size)
        
        if self.gaussian_std is None:
            #ne prendo il valore assoluto dipendentemente da np.min(img)
            gaussian_std = np.abs(np.min(img)*std_factor)

        if noise_mask is None:
            #scelgo una posizione random da cui iniziare
            noise_mask = np.random.normal(self.gaussian_mean, gaussian_std, size=self.input_size).astype('float32')

        if debug: print(noise_mask.shape)

        #genero uno spettrogramma di rumore bianco con distrib normale e lo sommo all'immagine
        img_with_noise = img + noise_mask

        return img_with_noise, noise_mask

if __name__ == "__main__":
    #TBC: vedi 1:Feature Extraction
    #https://github.com/mariostrbac/environmental-sound-classification/blob/main/notebooks/data_preprocessing.ipynb
    # build transformation pipelines for data augmentation for training phase and testing phase
    """
    transforms.Compose just clubs(raggruppa)all the transforms provided to it. 
    So, all the transforms in the transforms.Compose are applied to the input one by one.
    """
    training_transformations_pipelines = transforms.Compose([SpectogrammRightShift(input_size=128,width_shift_range=13,shift_probability=0.9),
                                                            SpectogramAddGaussNoise(input_size=128,prob_to_have_noise=0.55),
                                                            SpectogramReshape(output_size=(1,128,128))])

    testing_transformations_pipelines = transforms.Compose([SpectogramReshape(output_size=(1,128,128))])                                                  