import sys
sys.path.append('...')

import gc
import tensorflow.keras.backend as K
from .evaluation_strategy import NASEval
from .search_space import NASSearchSpace
from config import Params


class NASInterface(object):
    ''' 
        An interface that combines the Search Space + Evaluation Strategy 
        for the Artificial Bee Colony algorithm
    '''

    search_space = None
    eval_strategy = None

    def __init__(self, 
                 space_config=None,
                 eval_config=None):
        ''' Initializes the search space and evaluator '''

        space_config = space_config or Params.search_space_config()
        eval_config = eval_config or Params.evaluation_strategy_config()

        NASInterface.search_space = NASSearchSpace(space_config)
        NASInterface.eval_strategy = NASEval(eval_config)


    def sample(self):
        ''' Samples new random candidate architecture from the search space '''

        return NASInterface.search_space.sample()


    def evaluate(self, candidate):
        ''' Evaluates a given candidate architecture; returns loss value '''

        formatted = NASInterface.search_space.eval_format(candidate)
        res = NASInterface.eval_strategy.evaluate(formatted)

        # housekeeping
        K.clear_session()
        gc.collect()
        
        return res


    def get_neighbor(self, orig_arch):
        ''' Returns a random architecture with 1 op diff to the given candidate '''

        return NASInterface.search_space.get_neighbor(orig_arch)


    def fully_train_best_model(self, from_arch=True):
        '''
            Fully-train best-performing model
            (relies on paths set in Params)
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
            arch = formatted = NASInterface.search_space.eval_format(arch)
            return NASInterface.eval_strategy.fully_train(arch=arch)

        # check existence of weight file
        weight_file = os.path.join(Params.get_results_path(), Params['WEIGHT_FILES_SUBPATH'], weight_file)
        FileHandler.path_must_exist(weight_file)    # breaks if file does not exist
        

        # Continues training from saved h5 model (often results in lower fitness)
        return NASInterface.eval_strategy.fully_train(model_file=weight_file)

    
    def momentum_eval(self, candidate, weights_filename, m_epochs):
        ''' Trains a given network for additional m_epochs '''

        # check existence of weight file
        weights_path = os.path.join(Params.get_results_path(), Params['WEIGHT_FILES_SUBPATH'], weights_filename)
        FileHandler.path_must_exist(weights_path)    # breaks if file does not exist
        

        # Continues training from saved h5 model (often results in lower fitness)
        return NASInterface.eval_strategy.momentum_training(weights_path, m_epochs)


    @property 
    def is_minimize(self):
        ''' 
            Used by the ABC algorithm to determine whether this is 
            a minimization or maximization problem (we're maximizing accuracy)
        '''

        return False

