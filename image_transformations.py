import torch, torchvision
import copy

from torchvision import transforms
import numpy as np

#DATA AUGMENTATION per spettrogrammes
#Legenda:
#   TBC: To Be Canceled
class SpectrogramAddGaussNoise(object):
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

class SpectrogramShift(object):
    """Calcola il vettore dell' immagine dello spettrogramma shiftato"""
    #costruttore
    #TBC: input_size è il numero fisso dello spettrogramma (vedi 1 Feature Extraction)
    def __init__(self, input_size, width_shift_range, shift_prob=1.0, left = False, random_side = False):
        #checko che questi siano veri,altrimenti triggero subito errore

        #gli assert sono usati nel debugging, se sono sicuro
        #di cosa sono istanze , posso anche non metterli

        assert isinstance(input_size, tuple), "Input size must be a tuple (input_width, input_height)"
        self.input_size = input_size
        
        assert isinstance(width_shift_range, int) or isinstance(width_shift_range, float), "width_shift_range must be an int ora a float"
        if isinstance(width_shift_range,int):
            assert width_shift_range > 0 and width_shift_range <= self.input_size[1], "width_shift_range must be in range (0, input_size[1])"
            self.width_shift_range = width_shift_range
        else:
            #caso float
            assert width_shift_range > 0.0 and width_shift_range <= 1.0, "width_shift_range must be in range (0.0, 1.0)"
            self.width_shift_range = int(width_shift_range * self.input_size[1])

        assert isinstance(shift_prob, float), "shift_prob must be a float"
        assert shift_prob > 0.0 and shift_prob <= 1.0, "shift_prob must be in range (0, input_size[1])"
        self.shift_prob = shift_prob

        self.left = left
        self.random_side = random_side

    #image è un vettore numpy rappresentante lo spettrogramma
    #con questa l'istanza MyRightShift si comporterà come una '''funzione'''
    def __call__(self, img, shift_position = None, debug=False):
        if debug: print(shift_position)
        #se la shift prob è minore di una random non fare nulla sull'array
        if np.random.random() > self.shift_prob or shift_position==0:
            return img, 0
        
        #altrimenti mi creo una nuovo array pieno del valore minimo presente in quell array
        #i valori del vettore shiftato devono essere float32
        if debug: print(img.shape)
        img_shifted = np.full(self.input_size, np.min(img), dtype='float32')

        if shift_position is not None:
            assert isinstance(shift_position, int), "provided shift_position should be float"
        else:
            #scelgo una posizione random da cui iniziare
            shift_position = np.random.randint(1, self.width_shift_range)

            #(Optional) Left shift
            if self.left:
                shift_position *= -1
            
            #(Optional) Choose a random size
            if self.random_side and np.random.random() > 0.5:
                if shift_position<0: shift_position *= -1
            else:
                if shift_position>0: shift_position *= -1

        if debug: print(shift_position)

        #eseguo lo shift facendo deepcoopy dell'array in funzione di shift_position
        #in una porzione di image_shifted
        if shift_position<0:
            img_shifted[:,:shift_position] = copy.deepcopy(img[:,-shift_position:])
        else:
            img_shifted[:,shift_position:] = copy.deepcopy(img[:,:-shift_position])

        return img_shifted, shift_position

def SpectrogramReshape(object):

    def __init__(self, output_size):
        assert isinstance(output_size, tuple), "output_size must be a tuple (output_size_widt, output_size_height)"
        self.output_size = output_size

    def __call__(self, img):
      return img.reshape(self.output_size)



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