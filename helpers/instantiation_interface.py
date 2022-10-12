import sys
sys.path.append('...')

import re
import hashlib
import base64
import numpy as np
from functools import partial
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.datasets import cifar10, mnist
from tensorflow.keras.layers import Input, Conv2D, Add, Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.layers import SeparableConv2D, Dense, Dropout
from tensorflow.keras.layers import AveragePooling2D, BatchNormalization, ReLU



SS_CONFIG = {'depth': 5,
 'operations': {'sep5x5_128': partial(SeparableConv2D, filters=128, kernel_size=(5, 5), activation='relu', padding='same'),
  'sep3x3_128': partial(SeparableConv2D, filters=128, kernel_size=(3, 3), activation='relu', padding='same'),
  'sep5x5_64': partial(SeparableConv2D, filters=64, kernel_size=(5, 5), activation='relu', padding='same'),
  'sep3x3_64': partial(SeparableConv2D, filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
  'sep5x5_32': partial(SeparableConv2D, filters=32, kernel_size=(5, 5), activation='relu', padding='same'),
  'sep3x3_32': partial(SeparableConv2D, filters=32, kernel_size=(3, 3), activation='relu', padding='same'),
  'sep5x5_16': partial(SeparableConv2D, filters=16, kernel_size=(5, 5), activation='relu', padding='same'),
  'sep3x3_16': partial(SeparableConv2D, filters=16, kernel_size=(3, 3), activation='relu', padding='same'),
  'sep5x5_8': partial(SeparableConv2D, filters=8, kernel_size=(5, 5), activation='relu', padding='same'),
  'sep3x3_8': partial(SeparableConv2D, filters=8, kernel_size=(3, 3), activation='relu', padding='same'),
  'max_pool5x5': partial(MaxPooling2D, pool_size=(5, 5), strides=(1, 1), padding='same'),
  'avg_pool5x5': partial(AveragePooling2D, pool_size=(5, 5), strides=(1, 1), padding='same'),
  'max_pool3x3': partial(MaxPooling2D, pool_size=(3, 3), strides=(1, 1), padding='same'),
  'avg_pool3x3': partial(AveragePooling2D, pool_size=(3, 3), strides=(1, 1), padding='same'),
  'batch_norm': partial(BatchNormalization),
  'dropout': partial(Dropout, rate=0.2)},
 'residual_blocks_rate': 0.15}

EVAL_CONFIG = {'dataset': 'CIFAR10',
 'operations': SS_CONFIG['operations'],
 'output_stem': [partial(Flatten),
  partial(Dropout, rate=0.15),
  partial(Dense, units=1024, activation='relu'),
  partial(Dropout, rate=0.15),
  partial(Dense, units=512, activation='relu')],
 'input_stem': [partial(Conv2D, filters=32, kernel_size=(3, 3)),
  partial(BatchNormalization),
  partial(ReLU)],
 'epochs': 3,
 'full_train_epochs': 50,
 'lr': 0.001,
 'batch_size': 128,
 'optimizer': Adam,
 'termination_threshold_factor': 0.0,
 'termination_diminishing_factor': 0.25,
 'momentum_epochs': 25}



class NASSearchSpace(object):
    ''' Defines the Search Space used to sample candidates by HiveNAS '''
                    

    @staticmethod
    def sample():
        '''
            Samples a random point in the search space
        '''

        # assert self.all_paths != None, 'Search space needs to be initialized!'

        # idx = np.random.randint(0, len(self.all_paths))
        # return self.__encode_path(self.all_paths[idx])

        path = ['input']

        for l in range(SS_CONFIG['depth']):

            if np.random.rand() < SS_CONFIG['residual_blocks_rate']:
                sc_depth = np.random.randint(1, SS_CONFIG['depth'] - l + 1)
                path.append('L{}_sc_{}'.format(l+1, sc_depth))

            path.append('L{}_{}'.format(l+1, np.random.choice(
                list(SS_CONFIG['operations'].keys())
            )))
        
        path.append('output')

        return NASSearchSpace.__encode_path(path)


    @staticmethod
    def get_neighbor(path_str):
        ''' Returns a path with 1-op difference (a neighbor)'''

        path = NASSearchSpace.__strip_path(NASSearchSpace.__decode_path(path_str))

        component = np.random.randint(1, len(path) - 1)

        ops = []
        if path[component].startswith('sc'):
            # modify skip-connection (either remove it or change residual depth)
            sc_max_depth = len([op for op in path[component:] if not op.startswith('sc')])
            ops = [f'sc_{i}' for i in range(sc_max_depth)]
            ops.remove(path[component])
        else:
            # modify operation
            ops = list(SS_CONFIG['operations'].keys())
            ops.remove(path[component])
        
        # Replace randomly chosen component (operation) with any other op
        path[component] = np.random.choice(ops)

        # prune skip-connection if op == sc_0
        if path[component] == 'sc_0':
            del path[component]

        return NASSearchSpace.__encode_path(path)


    @staticmethod
    def eval_format(path):
        ''' 
            Formats a path for evaluation (stripped, decoded, and
            excluding input/output layers) given a string-encoded path
        '''

        return NASSearchSpace.__strip_path(NASSearchSpace.__decode_path(path))[1:-1]



    @staticmethod
    def __encode_path(path):
        ''' Returns a string encoding of a given path (list of ops)'''

        return '|'.join(NASSearchSpace.__strip_path(path))


    @staticmethod
    def __decode_path(path):
        ''' Returns a list of ops given a string-encoded path '''

        ops = path.split('|')

        for i in range(1, len(ops) - 1):
            ops[i] = 'L{}_{}'.format(i, ops[i])

        return ops


    @staticmethod
    def __strip_path(path):
        ''' Strips path of layer ID prefixes given a list of ops '''
        
        return [re.sub('L\d+_', '', s) for s in path]


    @staticmethod
    def compute_space_size():
        ''' 
            Returns the number of possible architectures in the given space
            (i.e operations and depth) for analytical purposes
        '''

        return len(list(SS_CONFIG['operations'].keys())) ** \
        SS_CONFIG['depth']




class NASEval(object):
    ''' Responsible for instantiating and evaluating candidate architectures '''


    @staticmethod
    def instantiate_network(arch):
        ''' Instantiates a Keras network given an architecture op list '''

        # residual counters
        res_count = []

        # add input according to given dataset shape
        if EVAL_CONFIG['dataset'] == 'CIFAR10':
            in_shape = (32,32,3)
        else:
            in_shape = (28,28)

        net = inputs = Input(shape=in_shape)

        # add input stem
        for op in EVAL_CONFIG['input_stem']:
            net = op()(net)

        # add hidden layers
        for layer in arch:

            if layer.startswith('sc'):
                # start residual block
                res_count.append((net, int(layer[3:])))
                continue

            assert layer in EVAL_CONFIG['operations'], f'Operation ({layer}) must be defined as a partial in HIVE_EVAL_CONFIG'
            net = EVAL_CONFIG['operations'][layer]()(net)

            for idx, row in enumerate(res_count):
                connection, counter = row
                counter -= 1

                # apply pooling to residual blocks to maintain shape
                # [deprecated] -- pooling layers padded
                # if 'pool' in layer:
                #     connection = self.config['operations'][layer]()(connection)

                if counter == 0:
                    # conv1x1 to normalize channels
                    fx = Conv2D(net.shape[-1], (1, 1), padding='same')(connection)
                    net = Add()([fx, net])
                    del res_count[idx]
                else:
                    res_count[idx] = (connection, counter)

        # add output stem
        for op in EVAL_CONFIG['output_stem']:
            net = op()(net)

        # add output layer
        net = Dense(10, activation='softmax')(net)

        return (Model(inputs, net), in_shape)


    @staticmethod
    def get_weights_filename(self, arch):
        ''' 
            Hashes the architecture op-list into a filename
        '''

        return hashlib.sha1(''.join(arch).encode("UTF-8")).hexdigest()


    @staticmethod
    def __compile_model(model):
        ''' Compiles model in preparation for evaluation '''

        model.compile(loss='sparse_categorical_crossentropy', \
                      optimizer=self.config['optimizer'](), \
                      metrics=['sparse_categorical_accuracy'])



''' Exposed API '''

def instantiate_network(arch_str):
    ''' Returns un-compiled model '''
    model, in_shape = NASEval.instantiate_network(NASSearchSpace.eval_format(arch_str))

    return (model, in_shape)


