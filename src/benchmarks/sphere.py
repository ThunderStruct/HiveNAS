import numpy as np
from .base import NumericalBenchmark

class Sphere(NumericalBenchmark):
    ''' Sphere optimization benchmark -- [ Σ Xi^2 , for i=1 in dim ] '''
    
    def __init__(self, dim, is_minimization=True):
        super(Sphere, self).__init__(dim, -100.0, 100.0, is_minimization)
        
    
    def evaluate(self, pos):
        ''' Evaluate a given position '''

        return {
            'fitness': sum(np.power(pos, 2)),
            'epochs': 1,
            'filename': '',
            'params': self.dim,
            'momentum': {'': ('', '')}
        }


    def momentum_eval(self, pos, weights, m_epochs):
        ''' Evaluates a given position for additional m_epochs '''

        return {
            'fitness': sum(np.power(pos, 2))
        }

        