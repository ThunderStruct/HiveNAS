
class FoodSource:
    
    def __init__(self, position=None, fitness=None):
        ''' 
            Data structure containing a FoodSource (position on the 
            optimization surface) and its fitness value
        '''

        self.pos = position
        self.fit = fitness
        self.eval_time = 0.0


    def encode_position(self):
        ''' Returns a stripped string-encoded position hash for dicts '''

        return str(self.pos).replace(' ', '')


    # --- Setters & Getters --- #

    @property
    def position(self):
        return self.pos
    

    @property
    def fitness(self):
        return self.fit


    @property
    def time(self):
        return self.eval_time


    @position.setter
    def position(self, value):
        self.pos = value


    @fitness.setter
    def fitness(self, value):
        self.fit = value


    @time.setter
    def time(self, value):
        self.eval_time = value


    def __str__(self):
        ''' For logging/debugging purposes '''

        return 'position: {}, fitness: {}, evaluation time: {}'.format(self.pos, 
                                                                       self.fit,
                                                                       self.eval_time)


    def __repr__(self):
        ''' For logging/debugging purposes '''

        return str(self)

