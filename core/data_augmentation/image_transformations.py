import torch, torchvision
import copy

from torchvision import transforms
import numpy as np

class SpectrogramAddGaussNoise(object):
 
    def __init__(self, input_size, gaussian_mean=0.0, gaussian_std=None, prob_to_have_noise=1.0):
        '''
        Add Gaussian Noise to the spectrogram using numpy
        Args:
          - input_size: size of the spectrogram
          OPTIONAL:
            - gaussian_mean: gaussian mean to apply
            - gaussian_std: gaussian standard deviation to apply
            - prob_to_have_noise: probability to have noise
        '''
        assert isinstance(input_size, tuple), "Input size must be a tuple (input_width, input_height)"
        self.input_size = input_size[:2]
        
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
        '''
        Add Gaussian Noise to the spectrogram using numpy
        Args:
          - img: input spectrogram
          OPTIONAL:
            - std_factor: float for standard deviation factor
            - noise_mask: noise to be applied
            - debug : boolean to set debug mode
        '''
        if np.random.random() > self.prob_to_have_noise <= 1.0:
            return img, np.zeros(self.input_size)
        
        if self.gaussian_std is None:
            gaussian_std = np.abs(np.min(img)*std_factor)

        if noise_mask is None:
            noise_mask = np.random.normal(self.gaussian_mean, gaussian_std, size=self.input_size).astype('float32')

        if debug: print(noise_mask.shape)

        #genero uno spettrogramma di rumore bianco con distrib normale e lo sommo all'immagine
        img_with_noise = img + noise_mask

        return img_with_noise, noise_mask

class SpectrogramShift(object):
    def __init__(self, input_size, width_shift_range, shift_prob=1.0, left = False, random_side = False):
        '''
        Callable that shift a spectrogram using numpy
        Args:
          - input_size: size of the spectrogram
          - width_shift_range: range of the shift
          OPTIONAL:
            - shift_prob: probability to have shifting
            - left: True if left shifting
            - random_side: True to decide randomly if left or right shift
        '''

        assert isinstance(input_size, tuple), "Input size must be a tuple (input_width, input_height)"
        self.input_size = input_size[:2]
        
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

    def __call__(self, img, shift_position = None, debug=False):
        '''
        Shifts the spectrogram using numpy
        Args:
          - img: input spectrogram
          OPTIONAL:
            - shift_position: shift amount
            - debug : boolean to set debug mode
        '''
        if debug: print(shift_position)

        if np.random.random() > self.shift_prob or shift_position==0:
            return img, 0
        
        if debug: print(img.shape)
        img_shifted = np.full(self.input_size, np.min(img), dtype='float32')

        if shift_position is not None:
            assert isinstance(shift_position, int), "provided shift_position should be float"
        else:
            shift_position = np.random.randint(1, self.width_shift_range)

            if self.left:
                shift_position *= -1
            
            if self.random_side and np.random.random() > 0.5:
                if shift_position<0: shift_position *= -1
            else:
                if shift_position>0: shift_position *= -1

        if debug: print(shift_position)

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



                                          