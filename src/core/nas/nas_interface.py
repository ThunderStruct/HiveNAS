"""The NAS Interface encapsulating the Evaluation Strategy and Search Space.
"""

import sys
sys.path.append('...')

import gc
import numpy as np
import tensorflow.keras.backend as K
from .evaluation_strategy import NASEval
from .nasbench_adapter import NASBench101Adapter
from .search_space import NASSearchSpace
from core.objective_interface import ObjectiveInterface
from config import Params


class NASInterface(ObjectiveInterface):
    '''
    An interface that combines the Search Space & Evaluation Strategy
    for the NAS Search Algorithm (ABC)

    Attributes:
        cls.eval_strategy (:class:`~core.nas.evaluation_strategy.NASEval`): NASEval instance used to instantiate and evaluate candidates
        cls.search_space (:class:`~core.nas.search_space.NASSearchSpace`): NASSearchSpace instance used to sample candidates and neighbors
    '''

    _ewc_enabled = False
    _ewc_importance = {}
    _ewc_lam = 0.5
    _ewc_sample_k = 4
    
    # Multi-objective settings
    _mo_enabled = False
    _mo_alpha = 1.0  # Weight for accuracy
    _mo_beta = 0.0   # Weight for parameters (penalty)
    _mo_params_scale = 1e6 # Scale params by this factor (e.g. 1M)

    def __init__(self,
                 space_config=None,
                 eval_config=None):
        '''Initializes the search space and evaluator

        Args:
            space_config (dict, optional): the predefined operational parameters pertaining to the search space (defined in :func:`~config.params.Params.search_space_config`)
            eval_config (dict, optional): the predefined operational parameters pertaining to evaluation (defined in :func:`~config.params.Params.evaluation_strategy_config`)
        '''

        space_config = space_config or Params.search_space_config()
        eval_config = eval_config or Params.evaluation_strategy_config()

        NASInterface.search_space = NASSearchSpace(space_config)

        # Use NASBench-101 adapter if USE_NASBENCH is enabled
        try:
            use_nasbench = Params['USE_NASBENCH']
        except KeyError:
            use_nasbench = False

        if use_nasbench:
            print("Using NASBench-101 for O(1) evaluation...")
            NASInterface.eval_strategy = NASBench101Adapter(eval_config, use_nasbench=True)
        else:
            NASInterface.eval_strategy = NASEval(eval_config)


    def sample(self):
        '''Samples new random candidate architecture from the search space 
        
        Returns:
            str: string-encoded representation of the sampled candidate architecture
        '''

        return NASInterface.search_space.sample()


    def evaluate(self, candidate):
        '''Evaluates a given candidate architecture
        
        Args:
            candidate (str): string-encoded representation of the architecture to be evaluated
        
        Returns:
            dict: a dictionary containing all relevant results to be saved, including: fitness, number of training epochs conducted (in case of ACT), hashed file name, number of trainable parameters, and the last epoch's momentum value if applicable
        '''

        formatted = NASInterface.search_space.eval_format(candidate)
        res = NASInterface.eval_strategy.evaluate(formatted)

        # Multi-Objective Scalarization
        if NASInterface._mo_enabled:
            acc = res['fitness']
            params = res.get('params', 0)
            
            # Normalize params (lower is better, so it's a penalty)
            # Fitness = Alpha * Accuracy - Beta * (Params / Scale)
            penalty = (params / NASInterface._mo_params_scale)
            
            # Apply EWC penalty if enabled (as an objective component)
            # Note: EWC is primarily used in get_neighbor, but we can add it here too
            # if we want the fitness to reflect adherence to the prior.
            # For now, we stick to Acc vs Params.
            
            # Store raw accuracy for reference
            res['raw_fitness'] = acc
            
            # Handle invalid architectures (low accuracy)
            if acc <= 0.101:  # Default invalid is 0.1
                 # Force very low fitness so they don't dominate due to 0 params
                 mo_fitness = -10.0
            else:
                 # Normal calculation
                 mo_fitness = (NASInterface._mo_alpha * acc) - (NASInterface._mo_beta * penalty)

            # Update fitness to be the scalarized value
            res['fitness'] = mo_fitness

        # housekeeping
        # K.clear_session()
        # gc.collect()  # Disabled for NASBench - causes 1.8s overhead per evaluation!

        return res


    def get_neighbor(self, orig_arch):
        '''Returns a random architecture with 1 op diff to the given candidate 
        
        Args:
            orig_arch (str): string-encoded representation of the candidate architecture
        
        Returns:
            str: string-encoded representation of the neighbor architecture
        '''

        if NASInterface._ewc_enabled:
            guided = self.__guided_neighbor(orig_arch)
            if guided is not None:
                return guided

        return NASInterface.search_space.get_neighbor(orig_arch)


    def fully_train_best_model(self, from_arch=True):
        '''
        Fully-train best-performing model
        (relies on paths set in :class:`~config.params.Params`)
        
        Args:
            from_arch (bool, optional): determines whether to train model from scratch \
            using the string representations of the architecture (:code:`from_arch = True`) \
            or load the saved model file and continue training (:code:`from_arch = False`). \
            \
            `Note: optimizer settings are typically not saved, \
            therefore training continuation from a model's file can result in a worse overall accuracy` \
            (`read more... <https://stackoverflow.com/a/58693088/3551916>`_).
        
        Returns:
            dict: a dictionary containing all relevant results to be saved, including: fitness, number of training epochs conducted (not including any previous trainings), hashed file name, number of trainable parameters
        
        Raises:
            :class:`FileNotFoundError`: raises an error if the results file or model h5 file (when applicable) do not exist
        '''

        # check existence of results file
        filename = f'{Params["CONFIG_VERSION"]}.csv'
        results_file = os.path.join(Params.get_results_path(), filename)
        FileHandler.path_must_exist(results_file)    # breaks if file does not exist

        # extract best fitness weight file
        results_df = pd.read_csv(results_file, header=0, index_col=0)
        weight_file = results_df.loc[results_df['fitness'] == results_df['fitness'].max(), 'weights_filename'].values[0]
        arch = results_df.loc[results_df['fitness'] == results_df['fitness'].max(), 'candidate'].values[0]

        print(f'\nFound best-performing model {{{arch}}} with a fitness score of {results_df["fitness"].max()}\n')
        
        # housekeeping -> results_df no longer needed and is potentially large
        del results_df
        gc.collect()

        if from_arch:
            # Retrains from scratch given the network arch
            arch = NASInterface.search_space.eval_format(arch)
            return NASInterface.eval_strategy.fully_train(arch=arch)

        # check existence of weight file
        weight_file = os.path.join(Params.get_results_path(), Params['WEIGHT_FILES_SUBPATH'], weight_file)
        FileHandler.path_must_exist(weight_file)    # breaks if file does not exist
        

        # Continues training from saved h5 model (often results in lower fitness)
        return NASInterface.eval_strategy.fully_train(model_file=weight_file)


    def train_custom_arch(self, arch_str):
        '''Fully trains a given string-encoded architecture (primarily used for debugging)
        
        Args:
            arch_str (str): string-encoded representation of the architecture

        Returns:
            dict: a dictionary containing all relevant results to be saved, \
            including: fitness, number of training epochs conducted (not including \
            any previous trainings), hashed file name, number of trainable parameters
        '''

        arch = NASInterface.search_space.eval_format(arch_str)
        print(f'Training {arch}...')
        
        return NASInterface.eval_strategy.fully_train(arch=arch)
    

    def momentum_eval(self, candidate, weights_filename, m_epochs):
        '''Trains a given network for additional :code:`m_epochs` 
        
        Args:
            candidate (str): string-encoded representation of the candidate architecture
            weights_filename (str): the SHA1-hashed unique string ID for the given architecture
            m_epochs (int): the additional momentum epochs the candidate should be trained for
        
        Returns:
            dict: final fitness value (accuracy) after training continuation
        '''

        # check existence of weight file
        weights_path = os.path.join(Params.get_results_path(), Params['WEIGHT_FILES_SUBPATH'], weights_filename)
        FileHandler.path_must_exist(weights_path)    # breaks if file does not exist
        

        # Continues training from saved h5 model (often results in lower fitness)
        return NASInterface.eval_strategy.momentum_training(weights_path, m_epochs)


    @property 
    def is_minimize(self):
        '''
        Used by the optimization algorithm to determine whether this is 
        a minimization or maximization problem
        
        Returns:
            bool: hard-coded :code:`False`; the search algorithm is always maximizing accuracy
        '''

        return False


    @classmethod
    def enable_multiobjective(cls, alpha=1.0, beta=0.05, params_scale=1e6):
        """Enable scalarized multi-objective evaluation (Acc - beta * Params)."""
        cls._mo_enabled = True
        cls._mo_alpha = alpha
        cls._mo_beta = beta
        cls._mo_params_scale = params_scale

    @classmethod
    def disable_multiobjective(cls):
        cls._mo_enabled = False

    @classmethod
    def enable_ewc_guidance(cls, importance=None, lam=0.5, sample_k=4):
        """Enable EWC-inspired neighbor guidance."""
        cls._ewc_enabled = True
        cls._ewc_importance = importance or {}
        cls._ewc_lam = lam
        cls._ewc_sample_k = max(1, int(sample_k))

    @classmethod
    def disable_ewc_guidance(cls):
        """Disable EWC guidance."""
        cls._ewc_enabled = False
        cls._ewc_importance = {}
        cls._ewc_lam = 0.5
        cls._ewc_sample_k = 4

    def __guided_neighbor(self, orig_arch):
        """Sample multiple neighbors and select one via EWC penalty."""
        if NASInterface.search_space is None:
            return None

        neighbors = set()
        attempts = 0
        max_attempts = max(5, NASInterface._ewc_sample_k * 5)

        while len(neighbors) < NASInterface._ewc_sample_k and attempts < max_attempts:
            neighbors.add(NASInterface.search_space.get_neighbor(orig_arch))
            attempts += 1

        if not neighbors:
            return None

        return self.__select_low_penalty_neighbor(orig_arch, list(neighbors))

    def __select_low_penalty_neighbor(self, current_arch, candidates):
        """Select neighbor with minimal importance-weighted changes."""
        if not candidates:
            return None

        if not NASInterface._ewc_importance:
            return candidates[0]

        current_ops = current_arch.split('|')[1:-1]
        best_idx = 0
        best_score = -np.inf

        for idx, candidate in enumerate(candidates):
            cand_ops = candidate.split('|')[1:-1]
            penalty = 0.0

            for pos in range(min(len(current_ops), len(cand_ops))):
                if current_ops[pos] != cand_ops[pos]:
                    penalty += NASInterface._ewc_importance.get(pos, 0.0)

            if len(cand_ops) > len(current_ops):
                extra = len(cand_ops) - len(current_ops)
                penalty += extra * max(NASInterface._ewc_importance.values(), default=0.0)

            score = -NASInterface._ewc_lam * penalty

            if score > best_score:
                best_score = score
                best_idx = idx

        return candidates[best_idx]
