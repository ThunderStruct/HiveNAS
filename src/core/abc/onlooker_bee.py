import sys
sys.path.append('...')

import time
import numpy as np
import pandas as pd
from .artificial_bee import ArtificialBee
from .food_source import FoodSource
from utils import Logger

class OnlookerBee(ArtificialBee):

    id_tracker = 0

    def __init__(self):
        super(OnlookerBee, self).__init__(None, OnlookerBee.id_tracker)
        OnlookerBee.id_tracker += 1
        self.employee = None


    def search(self, obj_interface):
        ''' 
            Exploit position (near a random EmployeeBee chosen according 
            to computed probability)
        '''

        self.food_source = self.get_random_neighbor(obj_interface)


    def assign_employee(self, employee):
        ''' Assigns an EmployeeBee to the Onlooker for neighbor-search '''

        self.employee = employee
        self.food_source = FoodSource(self.employee.food_source.position)

    
    def evaluate(self, obj_interface, itr):
        ''' 
            Evaluates sampled position and increments employee's trial counter
        '''

        Logger.evaluation_log('OnlookerBee', self.id, self.food_source.position)
        t = time.time()
        
        res = obj_interface.evaluate(self.food_source.position)
        # unpack
        self.food_source.fitness = res['fitness']
        epochs = res['epochs']
        weights_filename = res['filename']
        params = res['params']
        momentum = res['momentum']

        self.food_source.time = time.time() - t
        self.employee.trials += 1
        self.employee.greedy_select(self.food_source, obj_interface.is_minimize)

        series = pd.Series({
            'bee_type': type(self).__name__,
            'bee_id': self.id,
            'bee_parent': self.employee.id,
            'itr': itr,
            'candidate': self.food_source.position,
            'fitness': self.food_source.fitness,
            'center_fitness': self.get_center_fs().fitness,
            'momentum': sum([x[1] for _,x in momentum.items()]) / len(momentum),
            'epochs': epochs,
            'momentum_epochs': 0,
            'params': params,
            'weights_filename': weights_filename,
            'time': self.food_source.time
        })
        return series
        

    def get_center_fs(self):
        ''' Returns the parent's center food_source '''
        
        return self.employee.center_fs


    def __str__(self):
        ''' For logging/debugging purposes '''

        return 'OnlookerBee {} -> Parent Employee ({}) -- FS: {}'\
        .format(self.id, self.employee.id, self.food_source)

        