import numpy as np
import tensorflow as tf


class ImgAug:
    ''' 
        Element-wise image augmentation methods, used to preprocess a
        given dataset.
        (most affine transformations are already implemented in 
        Keras' ImageDataGenerator)
    '''

    @staticmethod
    def random_cutout(np_tensor):
        '''
            Randomly applies cutout augmentation to a given rank 3 tensor as
            defined in [1].

            [1] DeVries, T., & Taylor, G. W. (2017). Improved regularization of 
            convolutional neural networks with cutout.
        '''

        cutout_height = int(np.random.uniform(0.1, 0.2) * np_tensor.shape[0])
        cutout_width = int(np.random.uniform(0.1, 0.2) * np_tensor.shape[1])

        cutout_height_point = np.random.randint(np_tensor.shape[0] - cutout_height)
        cutout_width_point = np.random.randint(np_tensor.shape[1] - cutout_width)

        np_tensor[cutout_height_point: cutout_height_point + cutout_height, 
                  cutout_width_point: cutout_width_point + cutout_width, 
                  :] = 127    # 127 = grey cutout,
                              # 0 (black) or 255 (white) also valid
        
        return np.array(np_tensor)


    @staticmethod
    def random_contrast(np_tensor):
        ''' Apply random contrast augmentation '''
        
        return np.array(random_contrast(np_tensor, 0.5, 2))

    @staticmethod
    def random_saturation(np_tensor):
        ''' Apply random saturation augmentation '''

        return np.array(random_saturation(np_tensor, 0.2, 3))

    @staticmethod
    def augment(np_tensor):
        ''' Used by ImageDataGenerator's preprocess_function '''

        if np.random.uniform() <= Params['CONTRAST_AUG_PROB']:
            np_tensor = ImgAug.random_contrast(np_tensor)
        if np.random.uniform() <= Params['SATURATION_AUG_PROB']:
            np_tensor = ImgAug.random_saturation(np_tensor)
        if np.random.uniform() <= Params['CUTOUT_PROB']:
            np_tensor = ImgAug.random_cutout(np_tensor)

        return np_tensor

