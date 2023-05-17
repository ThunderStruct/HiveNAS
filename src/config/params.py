"""All operational parameters used by HiveNAS and configuration methods.
"""

import sys
sys.path.append('..')

import os
import yaml
from config import OperationCells
from utils import FileHandler
from functools import partial
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.optimizers.legacy import SGD
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.layers import SeparableConv2D, Dense, Dropout, Activation
from tensorflow.keras.layers import AveragePooling2D, BatchNormalization, ReLU
from tensorflow.keras.layers import GlobalAveragePooling2D


class Params:
    '''Wrapper for all global operational parameters
    and the configuration loader
    '''

    @staticmethod
    def config_form():
        '''Facilitates the configuration UI form (for Google Colab) \
        and exports all parameters as a dictionary
        
        Returns:
            dict: main global parameters dictionary (locals)
        '''


        ''' 
            Configuration Version (used as filename). 
            Can be considered the experiment setup's ID
        '''
        CONFIG_VERSION = 'hivenas_default'   #@param {type:"string"}

        ''' 
            Seed value used to reproduce results, -ve results will default to not specifying a seed 
            (may not be exact reproductions if the GPU backend is used) 
        '''
        SEED_VALUE = 42   #@param {type:"integer"}


        #@markdown ## ABC Optimizer Parameters
        #@markdown ---


        ''' Optimization problem (NAS or Numerical Benchmarks to test ABC) '''
        OPTIMIZATION_OBJECTIVE = 'NAS'  #@param ['NAS', 'Sphere_max', 'Sphere_min', 'Rosenbrock']

        ''' Max trials per Scout (i.e initial Food Source) '''
        ABANDONMENT_LIMIT = 3  #@param {type:"slider", min:1, max:50, step:1}

        ''' Number of bees in the colony (Employees + Onlookers) '''
        COLONY_SIZE = 7    #@param {type:"slider", min:1, max:50, step:1}

        ''' Distribution of Employees to Onlookers, resulting number of EmployeeBees = # of ScoutBees '''
        EMPLOYEE_ONLOOKER_RATIO = 0.43   #@param {type:"slider", min:0.1, max:1.0, step:0.05}

        ''' Number of ABC optimization iterations '''
        ITERATIONS_COUNT = 10    #@param {type:"slider", min:1, max:100, step:1}


        #@markdown \
        #@markdown ## File-Handling Parameters
        #@markdown ---


        ''' Save results every N evaluations (not iterations; iterations * colony_size) '''
        RESULTS_SAVE_FREQUENCY = 1   #@param {type:"slider", min:1, max:100, step:1}

        '''
            Result files base path (path will be created if it does not exist) 
            A local folder will be created after the CONFIG_VERSION
        '''
        RESULTS_BASE_PATH = '../res/archived results/'  #@param {type:"string"}

        ''' Training history files sub-path '''
        HISTORY_FILES_SUBPATH = 'training_history/'     #@param {type:"string"}

        ''' Enable weights saving for resumed training '''
        ENABLE_WEIGHT_SAVING = False    #@param {type:"boolean"}

        ''' Weight files sub-path (ensure that the path exists) '''
        WEIGHT_FILES_SUBPATH = 'weights/'    #@param {type:"string"}

        ''' Specifies whether or not to resume from existing results file (if exists)'''
        RESUME_FROM_RESULTS_FILE = False   #@param {type:'boolean'}


        #@markdown \
        #@markdown ## NAS Search Space Parameters
        #@markdown ---


        ''' -- NAS Search Space configuration -- '''

        #@markdown *( layers & hyperparameters must be defined as partial functions in code )*
        
        ''' Number of layers for sampled networks (excludes input/output stems) '''
        DEPTH = 5     #@param {type:"slider", min:1, max:10, step:1}

        ''' Search space operations '''
        OPERATIONS = {
            'search_space': [
                # 'conv3x3_64bnreluavgpool', 
                # 'resx2reg_32_conv3x3_64bnrelu', 
                # 'conv3x3_128bnreluavgpool', 
                # 'conv3x3_256bnreluavgpool', 
                # 'resx1reg_128_conv3x3_256bnrelu',
                # 'resx2reg_128_conv3x3_128bnrelu'

                'conv3x3_64bnreluavgpool', 
                'resx2reg_32_conv3x3_64bnrelu', 
                'conv3x3_128bnreluavgpool', 
                'conv3x3_256bnreluavgpool', 
                'resx1reg_128_conv3x3_256bnrelu',
                'resx1reg_128_conv3x3_128bnrelu'
            ],
            'reference_space': {
                'conv5x5_16': partial(Conv2D, filters=16, kernel_size=(5,5), activation='relu'),
                'conv3x3_16': partial(Conv2D, filters=16, kernel_size=(3,3), activation='relu'),
                'conv5x5_8': partial(Conv2D, filters=8, kernel_size=(5,5), activation='relu'),
                'conv3x3_8': partial(Conv2D, filters=8, kernel_size=(3,3), activation='relu'),
                'sep5x5_128': partial(SeparableConv2D, filters=128, kernel_size=(5,5), activation='relu', padding='same'),
                'sep3x3_128': partial(SeparableConv2D, filters=128, kernel_size=(3,3), activation='relu', padding='same'),
                'sep5x5_64': partial(SeparableConv2D, filters=64, kernel_size=(5,5), activation='relu', padding='same'),
                'sep3x3_64': partial(SeparableConv2D, filters=64, kernel_size=(3,3), activation='relu', padding='same'),
                'sep5x5_32': partial(SeparableConv2D, filters=32, kernel_size=(5,5), activation='relu', padding='same'),
                'sep3x3_32': partial(SeparableConv2D, filters=32, kernel_size=(3,3), activation='relu', padding='same'),
                'max_pool3x3': partial(MaxPooling2D, pool_size=(3,3), strides=(1,1), padding='same'),
                'avg_pool3x3': partial(AveragePooling2D, pool_size=(3,3), strides=(1,1), padding='same'),
                'global_avg_pool': partial(GlobalAveragePooling2D),
                'batch_norm': partial(BatchNormalization),
                'dropout': partial(Dropout, rate=0.15),
                'identity': partial(Activation, 'linear'),

                'conv3x3_32bnrelu': partial(OperationCells.ConvBnReLU, conv_kern=3, conv_filt=32),
                'conv3x3_16bnrelu': partial(OperationCells.ConvBnReLU, conv_kern=3, conv_filt=16),
                'conv3x3_256bnreluavgpool': partial(OperationCells.ConvBnReLUAvgPool, conv_kern=3, conv_filt=256),
                'conv3x3_128bnreluavgpool': partial(OperationCells.ConvBnReLUAvgPool, conv_kern=3, conv_filt=128),
                'conv3x3_64bnreluavgpool': partial(OperationCells.ConvBnReLUAvgPool, conv_kern=3, conv_filt=64),

                'resx2reg_128_conv3x3_256bnrelu': partial(OperationCells.ResidualConvBnReLU, conv_kern=3, conv_filt=256, reg_filters=128, block_count=2),
                'resx2reg_128_conv3x3_128bnrelu': partial(OperationCells.ResidualConvBnReLU, conv_kern=3, conv_filt=128, reg_filters=128, block_count=2),
                'resx2reg_128_conv3x3_64bnrelu': partial(OperationCells.ResidualConvBnReLU, conv_kern=3, conv_filt=64, reg_filters=128, block_count=2),
                'resx2reg_64_conv3x3_256bnrelu': partial(OperationCells.ResidualConvBnReLU, conv_kern=3, conv_filt=256, reg_filters=64, block_count=2),
                'resx2reg_64_conv3x3_128bnrelu': partial(OperationCells.ResidualConvBnReLU, conv_kern=3, conv_filt=128, reg_filters=64, block_count=2),
                'resx2reg_64_conv3x3_64bnrelu': partial(OperationCells.ResidualConvBnReLU, conv_kern=3, conv_filt=64, reg_filters=64, block_count=2),
                'resx2reg_32_conv3x3_256bnrelu': partial(OperationCells.ResidualConvBnReLU, conv_kern=3, conv_filt=256, reg_filters=32, block_count=2),
                'resx2reg_32_conv3x3_128bnrelu': partial(OperationCells.ResidualConvBnReLU, conv_kern=3, conv_filt=128, reg_filters=32, block_count=2),
                'resx2reg_32_conv3x3_64bnrelu': partial(OperationCells.ResidualConvBnReLU, conv_kern=3, conv_filt=64, reg_filters=32, block_count=2),
                'resx1reg_128_conv3x3_256bnrelu': partial(OperationCells.ResidualConvBnReLU, conv_kern=3, conv_filt=256, reg_filters=128, block_count=1),
                'resx1reg_128_conv3x3_128bnrelu': partial(OperationCells.ResidualConvBnReLU, conv_kern=3, conv_filt=128, reg_filters=128, block_count=1),
                'resx1reg_128_conv3x3_64bnrelu': partial(OperationCells.ResidualConvBnReLU, conv_kern=3, conv_filt=64, reg_filters=128, block_count=1),
                'resx1reg_64_conv3x3_256bnrelu': partial(OperationCells.ResidualConvBnReLU, conv_kern=3, conv_filt=256, reg_filters=64, block_count=1),
                'resx1reg_64_conv3x3_128bnrelu': partial(OperationCells.ResidualConvBnReLU, conv_kern=3, conv_filt=128, reg_filters=64, block_count=1),
                'resx1reg_64_conv3x3_64bnrelu': partial(OperationCells.ResidualConvBnReLU, conv_kern=3, conv_filt=64, reg_filters=64, block_count=1),
                'resx1reg_32_conv3x3_256bnrelu': partial(OperationCells.ResidualConvBnReLU, conv_kern=3, conv_filt=256, reg_filters=32, block_count=1),
                'resx1reg_32_conv3x3_128bnrelu': partial(OperationCells.ResidualConvBnReLU, conv_kern=3, conv_filt=128, reg_filters=32, block_count=1),
                'resx1reg_32_conv3x3_64bnrelu': partial(OperationCells.ResidualConvBnReLU, conv_kern=3, conv_filt=64, reg_filters=32, block_count=1),

            }
        }

        ''' Skip-Connections'/Residual Blocks' occurence rate (0.0 = disabled) '''
        STOCHASTIC_SC_RATE = 0.0    #@param {type:"slider", min:0.0, max:1.0, step:0.05}

        ''' Squeeze-Excitation blocks ratio (not implemented) '''
        # SE_RATIO = 0.0  #@param {type:"slider", min:0.0, max:0.25, step:0.25}


        #@markdown \
        #@markdown ## NAS Evaluation Strategy Parameters
        #@markdown ---


        ''' -- NAS Evaluation Strategy configuration -- '''

        ''' Dataset (classes/inputs are inferred internally) '''
        DATASET = 'CIFAR10'   #@param ["CIFAR10", "MNIST", "FASHION_MNIST"]

        ''' Static output stem, added to every candidate '''
        OUTPUT_STEM = [
            'global_avg_pool'
        ]

        ''' Static input stem, added to every candidate '''
        INPUT_STEM = [
            'conv3x3_32bnrelu'
        ]

        ''' Epochs count per candidate network '''
        EPOCHS = 7  #@param {type:"slider", min:1, max:25, step:1}
        
        ''' Momentum Augmentation epochs (0 = disabled ; overrides ENABLE_WEIGHT_SAVING) '''
        MOMENTUM_EPOCHS = 0 #@param {type:"slider", min:0, max:25}

        ''' Epochs count for the best performing candidate upon full training '''
        FULL_TRAIN_EPOCHS = 20 #@param {type:"slider", min:1, max:150, step:1}

        ''' 
            Threshold factor (beta) for early-stopping (refer to the TerminateOnThreshold class for details)
                1.0 = all networks will be terminated (minimum accuracy = 100%)
                0.0 = disable early-stopping, all networks will pass
                0.25 = for 10 classes, val_acc > 0.325 at epoch 1 will not be terminated
                       (tolerance decreased for every subsequent epoch)
        '''
        TERMINATION_THRESHOLD_FACTOR = 0.0 #@param {type:"slider", min:0.0, max:1.0, step:0.05}

        ''' Diminishing factor (zeta) for termination threshold over epochs '''
        TERMINATION_DIMINISHING_FACTOR = 0.25 #@param {type:"slider", min:0.1, max:1.0, step:0.05}

        ''' Initial learning rate for a LR scheduler (overrides lr defined in the OPTIMIZER param, 0.0 = disabled) '''
        INITIAL_LR = 0.08  #@param {type:"slider", min:0.0, max:0.1, step:0.001}

        ''' Final learning rate for a LR scheduler (overrides lr defined in the OPTIMIZER param, 0.0 = disabled) '''
        FINAL_LR = 0.01  #@param {type:"slider", min:0.0, max:0.1, step:0.001}

        ''' Batch size for every candidate evaluation '''
        BATCH_SIZE = 128     #@param {type:"slider", min:8, max:256, step:2}

        ''' Optimizer used for both NAS and full-training methods '''
        OPTIMIZER = partial(SGD, learning_rate=0.08, momentum=0.9, nesterov=True)


        #@markdown \
        #@markdown ## Data Augmentation Parameters
        #@markdown ---


        ''' 
            Enable affine transformations augmentation 
            (horizontal/vertical shifts, rotation, etc...)
        '''

        AFFINE_TRANSFORMATIONS_ENABLED = False   #@param {type:"boolean"}

        ''' Probability of random cutout augmentation occurence (0.0 = disabled) '''
        CUTOUT_PROB = 0.00    #@param {type:"slider", min:0.0, max:1.0, step:0.05}

        ''' Probability of random saturation augmentation occurence (0.0 = disabled) '''
        SATURATION_AUG_PROB = 0.00    #@param {type:"slider", min:0.0, max:1.0, step:0.05}

        ''' Probability of random contrast augmentation occurence (0.0 = disabled) '''
        CONTRAST_AUG_PROB = 0.00    #@param {type:"slider", min:0.0, max:1.0, step:0.05}

        # Parameters not defined as a dictionary to enable Colab Forms
        # Generally, locals() would introduce a security threat, however, 
        # in this controlled and undeloyable Colab environment, it's safe.
        return locals()


    ''' Main configuration dict '''
    __CONFIG = config_form.__func__()


    @staticmethod
    def init_from_yaml(path):
        '''Initializes the global parameters from a given yaml config file 
        
        Args:
            path (str): path to yaml configuration file
        
        '''

        def param_op_constructor(loader: yaml.SafeLoader, node: yaml.nodes.MappingNode):
            '''Constructs NAS Search Space operations (using the custom \
            yaml :code:`!Operation` tag)
            
            Args:
                loader (:class:`yaml.SafeLoader`): yaml default safe loader
                node (:class:`yaml.nodes.MappingNode`): yaml mapping node
            
            Returns:
                :class:`functools.partial`: partial function containing the neural operation
            '''

            # constructs an operation partial function from yaml !Operation tags
            op_dict = loader.construct_mapping(node)
            op = op_dict['op']
            del op_dict['op']

            return partial(globals()[op], **op_dict)
            
 
        def param_tuple_constructor(loader: yaml.SafeLoader, node: yaml.nodes.MappingNode):
            '''Constructs a tuple from the standard \
            :code:`tag:yaml.org,2002:python/tuple` (:code:`!!python/tuple`) yaml tag
            
            Args:
                loader (:class:`yaml.SafeLoader`): yaml default safe loader
                node (:class:`yaml.nodes.MappingNode`): yaml mapping node
            
            Returns:
                tuple: constructed tuple, \
                typically used to define kernel sizes/shapes in yaml
            '''

            # because for some reason we need an explicit tuple constructor

            return tuple(loader.construct_sequence(node))

        # register constructors
        loader = yaml.SafeLoader
        loader.add_constructor(u'tag:yaml.org,2002:python/tuple', param_tuple_constructor)
        loader.add_constructor('!Operation', param_op_constructor)

        config = FileHandler.load_yaml(path, loader)

        if not config:
            print(f'\nConfig file ({path}) is either invalid or does not exist.\n\n')
            return

        for key,val in config.items():
            if key not in Params.__CONFIG:
                # ensure config file keys are valid and match the hard-coded template
                print(f'\nConfig file ({path}) is invalid. Skipping item ({key})... \n\n')
                continue

            Params.__CONFIG[key] = val

        print(f'\nSuccessfully loaded the operational parameters from {path}.\n\n')


    @staticmethod
    def export_yaml(path, filename, from_formdata=False):
        '''Saves the current configurations to the given path as yaml 
        
        Args:
            path (str): output path to save the yaml config file to
            filename (str): output file name
            from_formdata (bool, optional): determines whether the export instruction \
            originated from the Google Colab UI form or called in code. *When it originates \
            from the form, data reload is required to ensure consistency (could be \
            altered within the form)*
        '''

        def param_op_representer(dumper, data):
            '''Serializes a partial function into the custom :code:`!Operation` \
            yaml tag
            
            Args:
                dumper (:class:`yaml.Dumper`): default pyyaml dumper
                data (partial): partial function data to be serialized
            
            Returns:
                :class:`yaml.nodes.MappingNode`: yaml mapping node representing \
                the operation
            '''

            # serialize partial functions into yaml !Operation
            serialized_data = {'op': data.func.__name__}
            serialized_data.update(data.keywords)
            
            return dumper.represent_mapping('!Operation', serialized_data, flow_style=True)

        def param_tuple_representer(dumper, data):
            '''Serializes a tuple into the :code:`tag:yaml.org,2002:python/tuple` \
            (:code:`!!python/tuple`) yaml tag
            
            Args:
                dumper (:class:`yaml.Dumper`): default pyyaml dumper
                data (tuple): tuple data to be serialized
            
            Returns:
                :class:`yaml.nodes.MappingNode`: yaml mapping node representing \
                the tuple
            '''

            # serialize tuples into yaml !!python/tuple

            return dumper.represent_sequence(u'tag:yaml.org,2002:python/tuple', data, flow_style=True)

        # register representers
        yaml.add_representer(tuple, param_tuple_representer)
        yaml.add_representer(partial, param_op_representer)
        yaml.Dumper.ignore_aliases = lambda *args : True

        # data source (changing the Colab form does not reflect on the main dict)
        data = Params.config_form() if from_formdata else Params.__CONFIG

        if FileHandler.export_yaml(data,
                                   path,
                                   filename):
            print(f'\nConfiguration file saved successfully to ({os.path.join(path, filename)})!\n\n')
        else:
            print('\nFailed to save config file!\n\n')


    @staticmethod
    def search_space_config():
        '''Returns the search space config dict 
        
        Returns:
            dict: dictionary containing the :class:`~core.nas.search_space.NASSearchSpace`-related \
            parameters
        '''

        res = {
            'depth': Params['DEPTH'],
            'operations': Params['OPERATIONS'],
            'stochastic_sc_rate': Params['STOCHASTIC_SC_RATE'],
        }

        return res
    

    @staticmethod
    def evaluation_strategy_config():
        '''Returns the evaluation strategy config dict 
        
        Returns:
            dict: dictionary containing the :class:`~core.nas.evaluation_strategy.NASEval`-related \
            parameters
        '''

        res = {
            'dataset': Params['DATASET'],
            'operations': Params['OPERATIONS'],
            'output_stem': Params['OUTPUT_STEM'],
            'input_stem': Params['INPUT_STEM'],
            'epochs': Params['EPOCHS'],
            'full_train_epochs': Params['FULL_TRAIN_EPOCHS'],
            'initial_lr': Params['INITIAL_LR'],
            'final_lr': Params['FINAL_LR'],
            'batch_size': Params['BATCH_SIZE'],
            'optimizer': Params['OPTIMIZER'],
            'termination_threshold_factor': Params['TERMINATION_THRESHOLD_FACTOR'],
            'termination_diminishing_factor': Params['TERMINATION_DIMINISHING_FACTOR'],
            'momentum_epochs': Params['MOMENTUM_EPOCHS']
        }

        return res


    @staticmethod
    def get_results_path():
        '''Gets the results path from :code:`RESULTS_BASE_PATH` and :code:`CONFIG_VERSION`
        
        Returns:
            str: the joined path to the results directory or :code:`None` if either \
            :code:`RESULTS_BASE_PATH` or :code:`CONFIG_VERSION` is invalid
        '''

        path = os.path.join(Params.__CONFIG['RESULTS_BASE_PATH'],
                            f'{Params.__CONFIG["CONFIG_VERSION"]}/')

        if FileHandler.validate_path(path):
            return path

        raise RuntimeError('Configuration overwriting denied; aborting...') 


    @staticmethod
    def get_all_config():
        '''Returns all operational parameters

        Returns:
            dict: returns the dict containing all configurations *(for \
            argparsing purposes)*
        '''

        return Params.__CONFIG


    @staticmethod
    def set_parameter(key, val):
        '''Overrides a default parameter (used by argparser)
        
        Args:
            key (str): dictionary key to select parameter
            val (any): new value to override default parameter
        '''

        if key not in Params.__CONFIG or not isinstance(val, type(Params.__CONFIG[key])):
            # invalid key or value type
            return False

        Params.__CONFIG[key] = val

        return True


    def __class_getitem__(cls, key):
        '''Subscript operator definition

        *Static class subscripting :code:`__class_getitem__` requires Python 3.7+*

        Used as :code:`Params['KEY']`
        
        Args:
            key (str): dictionary key to select parameter
        
        Returns:
            Any: subscripted parameter from the configuration dictionary
        '''
            
        return Params.__CONFIG[key]

