"""An barebone instantiation interface used by the analysis script

Attributes:
    EVAL_CONFIG (dict): the predefined operational parameters pertaining to the search space (defined in :func:`~config.params.Params.search_space_config`)
    SS_CONFIG (dict): the predefined operational parameters pertaining to evaluation (defined in :func:`~config.params.Params.evaluation_strategy_config`)
"""

import sys
sys.path.append('src')

from core.nas.nas_interface import NASInterface


''' Exposed API '''

INTERFACE_INSTANCE = NASInterface()

def instantiate_network(arch_str):
    '''Instantiates the network without compiling it (not needed for analysis purposes)
    
    Args:
        arch_str (str): string-encoded representation of the sampled candidate architecture
    
    Returns:
        (:class:`~tensorflow.keras.models.Model`, tuple): a tuple containing the \
        un-compiled Keras functional model from the given string-encoded architecture \
        and the model's input shape
    '''

    model = INTERFACE_INSTANCE.eval_strategy.instantiate_network(INTERFACE_INSTANCE.search_space.eval_format(arch_str))
    in_shape = INTERFACE_INSTANCE.eval_strategy.X_train.shape[1:4]

    return (model, in_shape)


