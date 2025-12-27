"""Particle Swarm Optimization for Neural Architecture Search.

This module implements PSO adapted for discrete architecture search spaces.
"""

import sys
sys.path.append('...')

import os
import time
import pandas as pd
import numpy as np
from config import Params
from utils import Logger, FileHandler


class ParticleSwarmOptimization:
    """Particle Swarm Optimization adapted for NAS.

    Attributes:
        swarm_size (int): Number of particles in the swarm
        obj_interface: The objective interface defining task boundaries
        results_df (DataFrame): Main results DataFrame containing all evaluated results
        w (float): Inertia weight
        c1 (float): Cognitive (personal best) coefficient
        c2 (float): Social (global best) coefficient
    """

    def __init__(self, obj_interface, swarm_size=10, w=0.7, c1=1.5, c2=1.5):
        """Initialize PSO algorithm.

        Args:
            obj_interface: ObjectiveInterface instance
            swarm_size (int): Number of particles
            w (float): Inertia weight
            c1 (float): Cognitive coefficient
            c2 (float): Social coefficient
        """
        self.obj_interface = obj_interface
        self.swarm_size = swarm_size
        self.w = w
        self.c1 = c1
        self.c2 = c2

        # Particle states
        self.particles = []  # Current positions (architectures)
        self.velocities = []  # Velocities (probabilities of change)
        self.pbest = []  # Personal best positions
        self.pbest_fitness = []  # Personal best fitnesses
        self.gbest = None  # Global best position
        self.gbest_fitness = None  # Global best fitness

    def _evaluate_particle(self, particle, itr=0):
        """Evaluate a particle (architecture).

        Args:
            particle (str): Architecture string
            itr (int): Current iteration

        Returns:
            pd.Series: Evaluation results
        """
        # Check if already evaluated
        if particle in self.results_df['candidate'].values:
            fitness = self.results_df[self.results_df['candidate'] == particle]['fitness'].values[0]
            return None, fitness

        # Evaluate new architecture
        eval_start = time.time()
        result = self.obj_interface.evaluate(particle)
        eval_time = time.time() - eval_start
        final_acc = result.get('final_acc', result['fitness'])

        series = pd.Series({
            'bee_type': 'Particle',
            'bee_id': len(self.results_df),
            'bee_parent': '-',
            'itr': float(itr),
            'candidate': particle,
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

    def _update_velocity(self, idx):
        """Update particle velocity (probability of changing each operation).

        For discrete space, velocity represents probability of exploration.

        Args:
            idx (int): Particle index
        """
        # Random coefficients
        r1, r2 = np.random.random(), np.random.random()

        # Update velocity (exploration probability)
        # Higher velocity = more likely to change architecture
        cognitive = self.c1 * r1 * (1.0 if self.pbest[idx] != self.particles[idx] else 0.0)
        social = self.c2 * r2 * (1.0 if self.gbest != self.particles[idx] else 0.0)

        self.velocities[idx] = self.w * self.velocities[idx] + cognitive + social

        # Clamp velocity to [0, 1]
        self.velocities[idx] = max(0.0, min(1.0, self.velocities[idx]))

    def _update_position(self, idx):
        """Update particle position based on velocity.

        Args:
            idx (int): Particle index
        """
        # With probability proportional to velocity, move towards pbest or gbest
        if np.random.random() < self.velocities[idx]:
            # Decide whether to move towards pbest or gbest
            if np.random.random() < 0.5 and self.pbest[idx] is not None:
                # Move towards personal best (get neighbor of pbest)
                self.particles[idx] = self.obj_interface.get_neighbor(self.pbest[idx])
            elif self.gbest is not None:
                # Move towards global best (get neighbor of gbest)
                self.particles[idx] = self.obj_interface.get_neighbor(self.gbest)
            else:
                # Explore randomly
                self.particles[idx] = self.obj_interface.sample()

    def optimize(self):
        """Main PSO optimization loop.

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

        # Initialize swarm
        Logger.status(0, f'Initializing swarm with {self.swarm_size} particles...')
        for i in range(self.swarm_size):
            particle = self.obj_interface.sample()
            self.particles.append(particle)
            self.velocities.append(0.5)  # Initial moderate velocity
            self.pbest.append(None)
            self.pbest_fitness.append(None)

        start_time = time.time()
        fitness_selector = max if not self.obj_interface.is_minimize else min

        # Main optimization loop
        try:
            from tqdm import tqdm
            import sys
            pbar = tqdm(range(Params['ITERATIONS_COUNT']),
                       desc="PSO Optimization",
                       ncols=100,
                       file=sys.stdout,
                       leave=False)
        except ImportError:
            pbar = range(Params['ITERATIONS_COUNT'])

        for itr in pbar:
            # Evaluate all particles
            if itr == 0:
                writer = pbar if hasattr(pbar, 'write') else None
                Logger.status(itr, 'Evaluating initial swarm positions...', writer=writer)

            for idx in range(self.swarm_size):
                _, fitness = self._evaluate_particle(self.particles[idx], itr)

                # Update personal best
                is_better = (fitness > self.pbest_fitness[idx] if not self.obj_interface.is_minimize
                            else fitness < self.pbest_fitness[idx]) if self.pbest_fitness[idx] is not None else True

                if is_better:
                    self.pbest[idx] = self.particles[idx]
                    self.pbest_fitness[idx] = fitness

                # Update global best
                is_global_better = (fitness > self.gbest_fitness if not self.obj_interface.is_minimize
                                   else fitness < self.gbest_fitness) if self.gbest_fitness is not None else True

                if is_global_better:
                    self.gbest = self.particles[idx]
                    self.gbest_fitness = fitness

            # Update velocities and positions for next iteration
            for idx in range(self.swarm_size):
                self._update_velocity(idx)
                self._update_position(idx)

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
