import sys
sys.path.append('...')

import time
import numpy as np
import pandas as pd
from .artificial_bee import ArtificialBee
from utils import Logger

class EmployeeBee(ArtificialBee):

    id_tracker = 0

    def __init__(self, food_source):
        super(EmployeeBee, self).__init__(food_source, EmployeeBee.id_tracker)
        EmployeeBee.id_tracker += 1
        self.trials = 0
        self.center_fs = food_source   # center_fs can be greedy-selected by
                                       # child onlookers (i.e the new
                                       # optimization center)


    def search(self, obj_interface):
        ''' 
            Explore new random position (near previously-sampled position)
        '''
        
        if self.food_source.fitness is None:
            # unevaluated (first iteration after abandonment reset)
            return

        # find neighbor near food_source in the bee's memory
        self.food_source = self.get_random_neighbor(obj_interface)


    def reset(self, new_fs):
        ''' Resets EmployeeBee once abandonment limit is reached '''

        self.trials = 0
        self.food_source = new_fs
        self.center_fs = new_fs


    def calculate_fitness(self):
        '''
            Calculate fitness of an EmployeeBee; given by:

                         ⎧ 1 / (1 + Fm(Xm→))       if  Fm(Xm→)≥0
            Fit_m(Xm→)=  ⎨
                         ⎩ 1 + abs(Fm(Xm→))        if  Fm(Xm→)<0
        '''

        fitm = 0
        if self.center_fs.fitness >= 0:
            fitm = 1 / (1 + self.center_fs.fitness)
        else:
            fitm = 1 + np.abs(self.center_fs.fitness)
        
        return fitm
    

    def compute_probability(self, sum_fitness):
        '''
            Calculate probability of an EmployeeBee being chosen by
            an OnlookerBee based on Fitess values; given by:

                Pn = Fit_n(Xn→) / [∑ (Fit_m(Xm→)) for all m]
        '''

        return self.calculate_fitness() / sum_fitness

    
    def evaluate(self, obj_interface, itr):
        ''' Evaluates sampled position and increments trial counter '''

        Logger.evaluation_log('EmployeeBee', self.id, self.food_source.position)
        t = time.time()

        res = obj_interface.evaluate(self.food_source.position)
        # unpack
        self.food_source.fitness = res['fitness']
        epochs = res['epochs']
        weights_filename = res['filename']
        params = res['params']
        momentum = res['momentum']

        self.food_source.time = time.time() - t
        # ACT early bandonment (ACT enabled and network could not pass epoch 1)
        abandon_early = Params['TERMINATION_THRESHOLD_FACTOR'] > 0.0 and epochs <= 1
        self.trials += 1 if not abandon_early else Params['ABANDONMENT_LIMIT']

        if self.center_fs.fitness is None:
            # Check if this evaluation is the first in its area 
            # (Iteration 1 after reset; i.e no need to greedy-select)
            self.center_fs = deepcopy(self.food_source)
        else:
            # Greedy select for iterations 2 ... Abandonment Limit
            self.greedy_select(self.food_source, obj_interface.is_minimize)

        # save data
        series = pd.Series({
            'bee_type': type(self).__name__,
            'bee_id': self.id,
            'bee_parent': '-',
            'itr': itr,
            'candidate': self.food_source.position,
            'fitness': self.food_source.fitness,
            'center_fitness': self.center_fs.fitness,
            'momentum': sum([x[1] for _,x in momentum.items()]),
            'epochs': epochs,
            'momentum_epochs': 0,
            'params': params,
            'weights_filename': weights_filename,
            'time': self.food_source.time
        })
        return series


    def greedy_select(self, n_food_source, is_minimize):
        ''' Update best FoodSource to minimize or maximize fitness '''

        if ((self.center_fs.fitness < n_food_source.fitness) and not is_minimize) or \
        ((self.center_fs.fitness > n_food_source.fitness) and is_minimize):
            self.center_fs.position = n_food_source.position
            self.center_fs.fitness = n_food_source.fitness
        

    def get_center_fs(self):
        ''' Returns the center food_source '''

        return self.center_fs
        

    def __str__(self):
        ''' For logging/debugging purposes '''

        return 'EmployeeBee {} -- FS: {}, trials: {}'.format(self.id, \
                                                             self.food_source, \
                                                             self.trials)
        
