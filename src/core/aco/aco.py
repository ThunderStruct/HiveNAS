"""Ant Colony Optimization for Neural Architecture Search.

This module implements ACO adapted for discrete architecture search spaces.
"""

import sys
sys.path.append('...')

import os
import time
import pandas as pd
import numpy as np
from config import Params
from utils import Logger, FileHandler


class AntColonyOptimization:
    """Ant Colony Optimization adapted for NAS.

    Attributes:
        colony_size (int): Number of ants
        obj_interface: The objective interface defining task boundaries
        results_df (DataFrame): Main results DataFrame containing all evaluated results
        alpha (float): Pheromone importance
        beta (float): Heuristic importance
        rho (float): Pheromone evaporation rate
        Q (float): Pheromone deposit factor
    """

    def __init__(self, obj_interface, colony_size=10, alpha=1.0, beta=2.0, rho=0.1, Q=1.0):
        """Initialize ACO algorithm.

        Args:
            obj_interface: ObjectiveInterface instance
            colony_size (int): Number of ants
            alpha (float): Pheromone importance
            beta (float): Heuristic importance
            rho (float): Evaporation rate
            Q (float): Pheromone deposit factor
        """
        self.obj_interface = obj_interface
        self.colony_size = colony_size
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q

        # Pheromone trails (architecture -> pheromone level)
        self.pheromones = {}
        self.initial_pheromone = 1.0

    def _get_pheromone(self, arch):
        """Get pheromone level for an architecture.

        Args:
            arch (str): Architecture string

        Returns:
            float: Pheromone level
        """
        if arch not in self.pheromones:
            self.pheromones[arch] = self.initial_pheromone
        return self.pheromones[arch]

    def _evaluate_ant(self, ant_arch, itr=0):
        """Evaluate an ant's architecture.

        Args:
            ant_arch (str): Architecture string
            itr (int): Current iteration

        Returns:
            tuple: (series, fitness)
        """
        # Check if already evaluated
        if ant_arch in self.results_df['candidate'].values:
            fitness = self.results_df[self.results_df['candidate'] == ant_arch]['fitness'].values[0]
            return None, fitness

        # Evaluate new architecture
        eval_start = time.time()
        result = self.obj_interface.evaluate(ant_arch)
        eval_time = time.time() - eval_start
        final_acc = result.get('final_acc', result['fitness'])

        series = pd.Series({
            'bee_type': 'Ant',
            'bee_id': len(self.results_df),
            'bee_parent': '-',
            'itr': float(itr),
            'candidate': ant_arch,
            'fitness': result['fitness'],
            'center_fitness': result['fitness'],
            'final_acc': final_acc,
            'epochs': result.get('epochs', 0),
            'params': result.get('params', 0),
            'weights_filename': result.get('filename', ''),
            'time': eval_time
        })

        self.results_df = pd.concat([self.results_df, series.to_frame().T], ignore_index=True)

        return series, result['fitness']

    def _construct_solution(self, best_archs):
        """Construct a solution using pheromone trails.

        Args:
            best_archs (list): List of best architectures found so far

        Returns:
            str: New architecture
        """
        # Use pheromone-based selection to decide exploration vs exploitation
        if len(best_archs) > 0 and np.random.random() < 0.7:
            # Exploit: Select from best architectures weighted by pheromones
            pheromones = [self._get_pheromone(arch) for arch in best_archs]
            total_pheromone = sum(pheromones)

            if total_pheromone > 0:
                probabilities = [p / total_pheromone for p in pheromones]
                selected = np.random.choice(best_archs, p=probabilities)
                # Get neighbor of selected architecture
                return self.obj_interface.get_neighbor(selected)

        # Explore: Sample new architecture
        return self.obj_interface.sample()

    def _update_pheromones(self, solutions, fitnesses):
        """Update pheromone trails.

        Args:
            solutions (list): List of architecture strings
            fitnesses (list): List of fitness values
        """
        # Evaporate pheromones
        for arch in self.pheromones:
            self.pheromones[arch] *= (1 - self.rho)

        # Deposit pheromones based on solution quality
        for arch, fitness in zip(solutions, fitnesses):
            # Deposit amount proportional to fitness
            deposit = self.Q * fitness if not self.obj_interface.is_minimize else self.Q * (1.0 - fitness)
            if arch in self.pheromones:
                self.pheromones[arch] += deposit
            else:
                self.pheromones[arch] = self.initial_pheromone + deposit

    def optimize(self):
        """Main ACO optimization loop.

        Returns:
            DataFrame: Results dataframe with all evaluated architectures
        """
        # Export configuration
        Params.export_yaml(Params.get_results_path(),
                          f'{Params["CONFIG_VERSION"]}.yaml')

        Logger.start_log()

        # Initialize results dataframe
        cols = ['bee_type', 'bee_id', 'bee_parent', 'itr', 'candidate',
                'fitness', 'final_acc', 'center_fitness', 'epochs', 'params',
                'weights_filename', 'time']
        self.results_df = pd.DataFrame(columns=cols)

        start_time = time.time()
        fitness_selector = max if not self.obj_interface.is_minimize else min

        # Track best architectures for exploitation
        best_archs = []
        best_fitness_threshold = 0.5  # Minimum fitness to be considered "good"

        # Main optimization loop
        try:
            from tqdm import tqdm
            import sys
            pbar = tqdm(range(Params['ITERATIONS_COUNT']),
                       desc="ACO Optimization",
                       ncols=100,
                       file=sys.stdout,
                       leave=False)
        except ImportError:
            pbar = range(Params['ITERATIONS_COUNT'])

        for itr in pbar:
            solutions = []
            fitnesses = []

            # Each ant constructs and evaluates a solution
            for ant_idx in range(self.colony_size):
                # Construct solution
                ant_arch = self._construct_solution(best_archs)
                solutions.append(ant_arch)

                # Evaluate solution
                _, fitness = self._evaluate_ant(ant_arch, itr)
                fitnesses.append(fitness)

                # Track good solutions
                if fitness > best_fitness_threshold and ant_arch not in best_archs:
                    best_archs.append(ant_arch)

            # Update pheromone trails
            self._update_pheromones(solutions, fitnesses)

            # Keep best architectures list manageable
            if len(best_archs) > 20:
                # Keep top 20 by fitness
                best_archs_with_fitness = [(arch, self.results_df[self.results_df['candidate'] == arch]['fitness'].values[0])
                                           for arch in best_archs]
                best_archs_with_fitness.sort(key=lambda x: x[1], reverse=not self.obj_interface.is_minimize)
                best_archs = [arch for arch, _ in best_archs_with_fitness[:20]]

            # Update progress bar
            if hasattr(pbar, 'set_postfix'):
                best_fitness = fitness_selector(self.results_df['fitness'].tolist())
                pbar.set_postfix({'best_fitness': f'{best_fitness:.4f}', 'evals': len(self.results_df)})

            # Status logging
            if itr % Params['RESULTS_SAVE_FREQUENCY'] == 0:
                best_fitness = fitness_selector(self.results_df['fitness'].tolist())
                writer = pbar if hasattr(pbar, 'write') else None
                Logger.status(itr,
                              f'Best fitness: {best_fitness}, Total time (s): {time.time() - start_time}',
                              writer=writer)

        Logger.end_log()

        return self.results_df
