from .food_source import FoodSource
from abc import ABC, abstractmethod

class ArtificialBee(ABC):
    ''' 
        Abstract class for Employee & Onlooker Bees
    '''

    def __init__(self, food_source, id):
        self.food_source = food_source
        self.id = id if id is not None else -1
    

    def get_random_neighbor(self, obj_interface):
        '''
            Finds a random neighbor in the vicinity of the parent 
            Parent(Onlooker) = Employee,
            Parent(Employee) = Scout
            Given by [1]:

                Xmi = Li + rand(0, 1) ∗ (Ui − Li)   => Initial FoodSource
                                                       (Scout)
                υmi = Xmi + ϕmi(Xmi − Xki)          => Neighboring FoodSource 
                                                       (Employee/Onlooker)

                Where υmi is a neighboring FoodSource
                (definition of "neighboring" given in [2]; 
                TLDR - in numerical and continuous optimization problems, 
                a dimensional component is incremented/decremented. 
                In NAS context, it is a 1-operation difference per network)


            [1] Karaboga, D., & Basturk, B. (2007). A powerful and efficient 
            algorithm for numerical function optimization: artificial bee 
            colony (ABC) algorithm. Journal of global optimization, 39(3), 
            459-471.

            [2] White, C., Nolen, S., & Savani, Y. (2021, December). 
            Exploring the loss landscape in neural architecture search. 
            In Uncertainty in Artificial Intelligence (pp. 654-664). PMLR.
        '''
        
        pos = self.get_center_fs().position
        neighbor_pos = obj_interface.get_neighbor(pos)

        return FoodSource(neighbor_pos)


    def is_evaluated(self):
        ''' 
            Checks if food source is evaluated for
            solution tracking purposes
        '''

        return self.food_source is not None and \
        self.food_source.fitness is not None

    @abstractmethod
    def get_center_fs(self):
        raise NotImplementedError()


    @abstractmethod
    def evaluate(self, obj_interface):
        raise NotImplementedError()


    @abstractmethod
    def search(self, obj_interface):
        raise NotImplementedError()


    def __repr__(self):
        ''' For logging/debugging purposes '''
        return str(self)

        