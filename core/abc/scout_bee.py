import sys
sys.path.append('...')

from .food_source import FoodSource
from config import Params

class ScoutBee:
    ''' 
        Scout Bees static class, 
        responsible for initializing random FoodSources position
        and reseting employed bees when abandonment limit is reached
    '''


    @staticmethod
    def sample(obj_interface):
        ''' Sample a random point from the objective function '''

        return FoodSource(obj_interface.sample())


    @staticmethod
    def check_employee_trials(employees, obj_interface):
        ''' Check if EmployeeBees reached abandonment limit '''

        for employee in employees:
            if employee.trials >= Params['ABANDONMENT_LIMIT']:
                # Reset EmployeeBees to a new ScoutBee position
                employee.reset(ScoutBee.sample(obj_interface))

        