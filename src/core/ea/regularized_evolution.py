import pandas as pd
import numpy as np
import time
import random
from config import Params
from utils import Logger

class RegularizedEvolution:
    """Regularized Evolution (Realized Evolution) for NAS.
    
    Based on: Real, E., Aggarwal, A., Huang, Y., & Le, Q. V. (2019). 
    Regularized evolution for image classifier architecture search. 
    AAAI.
    """

    def __init__(self, obj_interface, population_size=100, sample_size=25, cycles=5000):
        """
        Args:
            obj_interface: ObjectiveInterface instance
            population_size (int): Size of the active population (history)
            sample_size (int): Tournament size
            cycles (int): Number of evolution cycles (evaluations)
        """
        self.obj_interface = obj_interface
        self.population_size = population_size
        self.sample_size = sample_size
        self.cycles = cycles
        
        # Population is a list of dicts: {'arch': str, 'fitness': float}
        self.population = [] 
        self.history = []
        self.results_df = None

    def _mutate(self, parent_arch):
        """Mutate architecture (get neighbor)."""
        return self.obj_interface.get_neighbor(parent_arch)

    def optimize(self):
        """Main Regularized Evolution loop."""
        Logger.start_log()
        
        cols = ['itr', 'candidate', 'fitness', 'final_acc', 'epochs', 'params', 'time']
        results_data = []
        
        start_time = time.time()
        
        try:
            from tqdm import tqdm
            import sys
            pbar = tqdm(range(self.cycles),
                        desc="Evolution",
                        ncols=100,
                        file=sys.stdout,
                        leave=False)
        except ImportError:
            pbar = range(self.cycles)

        eval_count = 0
        
        # Initialize population with random architectures
        while len(self.population) < self.population_size and eval_count < self.cycles:
            arch = self.obj_interface.sample()
            
            eval_start = time.time()
            res = self.obj_interface.evaluate(arch)
            eval_time = time.time() - eval_start
            
            fitness = res['fitness']
            final_acc = res.get('final_acc', fitness)
            
            # Store individual
            individual = {'arch': arch, 'fitness': fitness}
            self.population.append(individual)
            self.history.append(individual)
            
            results_data.append({
                'itr': eval_count,
                'candidate': arch,
                'fitness': fitness,
                'final_acc': final_acc,
                'epochs': res.get('epochs', 0),
                'params': res.get('params', 0),
                'time': eval_time
            })
            
            eval_count += 1
            if hasattr(pbar, 'update'):
                pbar.update(1)

        # Evolution loop
        while eval_count < self.cycles:
            # 1. Sample subset (Tournament Selection)
            # In Regularized Evolution, we remove the OLDEST, not the worst.
            # But selection is from the current active population.
            sample = random.sample(self.population, min(len(self.population), self.sample_size))
            
            # 2. Select best parent
            parent = max(sample, key=lambda x: x['fitness'])
            
            # 3. Mutate
            child_arch = self._mutate(parent['arch'])
            
            # 4. Evaluate
            eval_start = time.time()
            res = self.obj_interface.evaluate(child_arch)
            eval_time = time.time() - eval_start
            
            fitness = res['fitness']
            final_acc = res.get('final_acc', fitness)
            
            child = {'arch': child_arch, 'fitness': fitness}
            
            # 5. Update Population (Regularization: Kill oldest)
            self.population.append(child)
            self.history.append(child)
            
            if len(self.population) > self.population_size:
                self.population.pop(0) # Remove oldest
                
            results_data.append({
                'itr': eval_count,
                'candidate': child_arch,
                'fitness': fitness,
                'final_acc': final_acc,
                'epochs': res.get('epochs', 0),
                'params': res.get('params', 0),
                'time': eval_time
            })
            
            eval_count += 1
            
            # Update progress bar
            if hasattr(pbar, 'set_postfix'):
                best_fitness = max(x['fitness'] for x in self.history)
                pbar.set_postfix({'best': f'{best_fitness:.4f}'})
            if hasattr(pbar, 'update'):
                pbar.update(1)

        Logger.end_log()
        
        self.results_df = pd.DataFrame(results_data, columns=cols)
        return self.results_df

