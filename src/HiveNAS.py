"""Top-level module used to run the framework.
"""

import os
# import plaidml.keras
from config import Params
from utils import Logger
from utils import ArgParser
from benchmarks import Sphere, Rosenbrock
from core import NASInterface, ArtificialBeeColony


# plaidml.keras.install_backend()
# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"


class HiveNAS:
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

        if Params['OPTIMIZATION_OBJECTIVE'] != 'NAS':
            raise ValueError('Attempting to optimize a neural topology for a non-NAS objective')

        objective_interface = NASInterface()
        abc = ArtificialBeeColony(objective_interface)

        abc.optimize()

    
    @staticmethod
    def test_numerical_optimization(evaluation_logging=True,
                                    config_path=None):
        '''Used to test Artificial Bee Colony's optimization on numerical benchmarks
        
        Raises:
            :class:`ValueError`: raised when the :func:`~HiveNAS.test_numerical_optimization` \
            is called while :code:`OPTIMIZATION_OBJECTIVE` parameter is improperly set`
        
        Args:
            evaluation_logging (bool, optional): determines whether to log \
            evaluation info or not; defaults to :code:`True`
            config_path (str, optional): yaml configuration file path; \
            defaults to hard-coded config in :class:`~config.params.Params`
            kill_after (bool, optional): kills the Colab runtime after completion \
            to preserve computational units and free the instance for others to use
        '''

        if config_path:
            # load yaml config from given path
            Params.init_from_yaml(config_path)

        Logger.EVALUATION_LOGGING = evaluation_logging
        
        if Params['OPTIMIZATION_OBJECTIVE'] == 'Sphere_min':
            objective_interface = Sphere(10)
        elif Params['OPTIMIZATION_OBJECTIVE'] == 'Sphere_max':
            objective_interface = Sphere(10, False)
        elif Params['OPTIMIZATION_OBJECTIVE'] == 'Rosenbrock':
            objective_interface = Rosenbrock(2)
        else:
            raise ValueError('Failed to optimize numerical benchmark. \
            Please ensure that the OPTIMIZATION_OBJECTIVE parameter is set accordingly.')

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


    @staticmethod
    def set_reproducible(seed_value):
        '''Sets the backend's RNG seed to reproduce results.

        Note: Keras has additional internal stochastic processes when using \
        GPU acceleration. Run the framework a couple of times and you're bound to \
        get an exact reproduction of the rseults.

        Args:
            seed_value (int): the RNG seed value, According to :class:`~config.Params`, \
            negative values disable reproductions and revert to default randomness
        '''

        if seed_value >= 0:
            os.environ['PYTHONHASHSEED'] = str(seed_value)
            random.seed(seed_value)
            np.random.seed(seed_value)

            import tensorflow.compat.v1 as tfv1
            tfv1.set_random_seed(seed_value)
            session_conf = tfv1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
            sess = tfv1.Session(graph=tfv1.get_default_graph(), config=session_conf)
            tfv1.keras.backend.set_session(sess)


# Run HiveNAS
if __name__ == '__main__':

    args = ArgParser.get_arguments(Params.get_all_config().items())

    # set config args
    for key, val in args.items():
        if key.upper() in Params.get_all_config():
            Params.set_parameter(key.upper(), val)
                
    # run HiveNAS
    if 'fully-train' in args and args['fully-train']:
        HiveNAS.fully_train_topology(args['config_file'])
    elif 'evaluate-arch' in args and args['evaluate-arch']:
        HiveNAS.manual_arch_evaluation(args['evaluate-arch'], args['config_file'])
    else:
        HiveNAS.find_topology(args['verbose'], args['config_file'])

    