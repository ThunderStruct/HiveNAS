"""
Random Search baseline for Neural Architecture Search.

This provides a simple baseline that randomly samples architectures
for comparison with swarm intelligence and RL methods.
"""

import time
import pandas as pd
from config import Params
from utils.logger import Logger


class RandomSearch:
    """Random Search optimizer for NAS."""

    def __init__(self, obj_interface, budget=5000):
        """
        Initialize Random Search.

        Args:
            obj_interface: NASInterface instance
            budget (int): Total number of evaluations
        """
        self.obj_interface = obj_interface
        self.budget = budget
        self.results_df = None

    def optimize(self):
        """
        Run random search.

        Returns:
            pandas.DataFrame: Results containing all evaluated architectures
        """
        Logger.start_log()

        # Initialize results storage
        cols = ['itr', 'candidate', 'fitness', 'final_acc', 'epochs', 'params', 'time']
        self.results_df = pd.DataFrame(columns=cols)

        best_fitness = None
        start_time = time.time()

        # Progress bar
        try:
            from tqdm import tqdm
            import sys
            pbar = tqdm(range(self.budget),
                       desc="Random Search",
                       ncols=100,
                       file=sys.stdout,
                       leave=False)
        except ImportError:
            pbar = range(self.budget)

        for itr in pbar:
            # Sample random architecture
            candidate = self.obj_interface.sample()

            # Evaluate
            eval_start = time.time()
            result = self.obj_interface.evaluate(candidate)
            eval_time = time.time() - eval_start

            # Extract fitness
            if isinstance(result, dict):
                fitness = result['fitness']
                final_acc = result.get('final_acc', fitness)  # Get final accuracy if available
                epochs = result.get('epochs', 0)
                params = result.get('params', 0)
            else:
                fitness = result
                final_acc = fitness
                epochs = 0
                params = 0

            # Track best
            if best_fitness is None:
                best_fitness = fitness
            elif not self.obj_interface.is_minimize and fitness > best_fitness:
                best_fitness = fitness
            elif self.obj_interface.is_minimize and fitness < best_fitness:
                best_fitness = fitness

            # Save result
            result_row = pd.DataFrame([{
                'itr': itr,
                'candidate': candidate,
                'fitness': fitness,
                'final_acc': final_acc,
                'epochs': epochs,
                'params': params,
                'time': eval_time
            }])
            self.results_df = pd.concat([self.results_df, result_row], ignore_index=True)

            # Update progress bar
            if hasattr(pbar, 'set_postfix'):
                pbar.set_postfix({
                    'best_fitness': f'{best_fitness:.4f}',
                    'current': f'{fitness:.4f}'
                })

            # Print status every 10% of budget
            if (itr + 1) % max(1, (self.budget // 10)) == 0:
                elapsed = time.time() - start_time
                writer = pbar if hasattr(pbar, 'write') else None
                Logger.status(
                    itr,
                    f'Best fitness: {best_fitness}, Total time (s): {elapsed}',
                    writer=writer
                )

        Logger.end_log()

        return self.results_df
