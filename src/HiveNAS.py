"""Top-level module used to run the framework.
"""

import os
# import plaidml.keras
from functools import partial
from argparse import ArgumentParser
from config import Params
from utils import Logger
from benchmarks import Sphere, Rosenbrock
from core import NASInterface, ArtificialBeeColony


# plaidml.keras.install_backend()
# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"


class HiveNAS(object):
    '''Encapsulates all high level modules and runs the ABC-based optimization
    '''

    @staticmethod
    def find_topology(evaluation_logging=True,
                      config_path=None):
        '''Runs the base NAS optimization loop 
        
        Args:
            evaluation_logging (bool, optional): determines whether to log \
            evaluation info or not; defaults to :code:`True`
            config_path (str, optional): yaml configuration file path; \
            defaults to hard-coded config in :class:`~config.params.Params`
        '''

        if config_path:
            # load yaml config from given path
            Params.init_from_yaml(config_path)

        Logger.EVALUATION_LOGGING = evaluation_logging

        if Params['OPTIMIZATION_OBJECTIVE'] == 'NAS':
            objective_interface = NASInterface()     
        elif Params['OPTIMIZATION_OBJECTIVE'] == 'Sphere_min':
            objective_interface = Sphere(10)
        elif Params['OPTIMIZATION_OBJECTIVE'] == 'Sphere_max':
            objective_interface = Sphere(10, False)
        elif Params['OPTIMIZATION_OBJECTIVE'] == 'Rosenbrock':
            objective_interface = Rosenbrock(2)

        abc = ArtificialBeeColony(objective_interface)

        abc.optimize()

    
    @staticmethod
    def fully_train_topology(config_path=None):
        '''Given the current configuration file, 
        extract the best previously-found topology and fully-train it 
        
        Args:
            config_path (str, optional): yaml configuration file path; \
            defaults to hard-coded config in :class:`~config.params.Params`
        '''

        if config_path:
            # load yaml config from given path
            Params.init_from_yaml(config_path)

        Logger.start_log()

        # loads architecture and optimizes its weights over a larger number of epochs; 
        # from_arch sepcifies whether to re-instantiate the network and train from scratch 
        # or resume training from weights file
        res = NASInterface().fully_train_best_model(from_arch=True)

        Logger.end_log()

        print(res)


    @staticmethod
    def manual_arch_evaluation(arch_str,
                               config_path=None):
        '''Evaluates a given architecture string (used primarily for debugging)

        Args:
            config_path (str, optional): yaml configuration file path; \
            defaults to hard-coded config in :class:`~config.params.Params`
            arch_str (str): string-encoded representation of the architecture to evaluate
        '''

        if config_path:
            # load yaml config from given path
            Params.init_from_yaml(config_path)

        Logger.start_log()

        # loads architecture and optimizes its weights over a larger number of epochs; 
        # from_arch sepcifies whether to re-instantiate the network and train from scratch 
        # or resume training from weights file
        res = NASInterface().train_custom_arch(arch_str=arch_str)

        Logger.end_log()

        print(res)


# Run HiveNAS
if __name__ == "__main__":

    # parse arguments
    parser = ArgumentParser()

    parser.add_argument('-ea', '--evaluate-arch',
                        type=bool,
                        help='Manually evaluate an architecture (string-encoded)',
                        default=None)
    parser.add_argument('-ft', '--fully-train',
                        type=bool,
                        help='Specifies whether to fully-train the best \
                        candidate or perform the initial shallow NAS',
                        choices=[True, False],
                        default=False)
    parser.add_argument('-vb', '--verbose',
                        help='Specifies whether to log all evaluation details',
                        default=False,
                        action='store_true')
    parser.add_argument('-c', '--config-file',
                        type=str,
                        help='Configuration file (relative) path',
                        default=None)

    abbrevs = []
    for key, val in Params.get_all_config().items():
        # TODO: add :code:`help` argument to generated args list
        if not isinstance(val, list) and not isinstance(val, dict) and not isinstance(val, partial):
            split_ls = key.lower().split('_')
            abbrev = '-' + ''.join([w[0] for w in split_ls])
            param = '--' + '-'.join(split_ls)

            if abbrev in abbrevs:
                # handle abbreviation conflicts
                abbrev = f'-{split_ls[0][0]}{split_ls[0][int(len(split_ls[0])/2)+1]}'

            parser.add_argument(abbrev, param, default=val, type=type(val))
            abbrevs.append(abbrev)


    args = parser.parse_args()
    args = vars(args)

    # set config args
    for key, val in args.items():
        if key.upper() in Params.get_all_config():
            Params.set_parameter(key.upper(), val)
                
    # run HiveNAS
    if args['fully-train']:
        HiveNAS.fully_train_topology(args['verbose'], args['config_file'])
    elif args['evaluate-arch']:
        HiveNAS.manual_arch_evaluation(args['evaluate-arch'], args['config_file'])
    else:
        HiveNAS.find_topology(args['config_file'])

    