import numpy as np
from abc import ABC, abstractmethod

class NumericalBenchmark(ABC):
    ''' Abstract class for Numerical Optimization Benchmarks '''

    def __init__(self, dim, minv, maxv, minimization):
        self.dim = dim
        self.minv = minv
        self.maxv = maxv
        self.minimization = minimization


    def sample(self):
        ''' Samples a random point from the objective function '''

        return np.random.uniform(low=self.minv, high=self.maxv, \
                                 size=self.dim)
        

    def get_neighbor(self, pos):
        ''' Finds a random neighbor by nudging 1 component of the pos by phi '''

        op = np.random.choice(pos)
        phi = np.random.uniform(low=-1, high=1, size=len(pos))
        neighbor_pos = pos + (pos - op) * phi

        return self.__eval_boundary(neighbor_pos)
    

    def __eval_boundary(self, n_pos):
        '''
            Ensures the newly sampled position (neighbor) lies within the
            Search Space boundaries
        '''

        if (n_pos < self.minv).any() or \
        (n_pos > self.maxv).any():
            n_pos[n_pos < self.minv] = self.minv
            n_pos[n_pos > self.maxv] = self.maxv
        return n_pos


    @property
    def minimum(self):
        return self.minv


    @property
    def maximum(self):
        return self.maxv


    @property 
    def is_minimize(self):
        return self.minimization


    @abstractmethod
    def evaluate(self, pos):
        raise NotImplementedError()

