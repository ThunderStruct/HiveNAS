from .base import NumericalBenchmark
from scipy import optimize

class Rosenbrock(NumericalBenchmark):
    ''' 
        -- Rosenbrock optimization benchmark -- 
        [ Σ { 100(Xi+1 - Xi)^2 + (Xi - 1)^2 } , for i=1 in dim - 1 ] 
    '''

    def __init__(self, dim):
        super(Rosenbrock, self).__init__(dim, -30.0, 30.0, True)


    def evaluate(self, pos):
        ''' Scipy implementation '''

        return optimize.rosen(pos), 1, ''

