"""Simple policy-gradient inspired controller for NAS benchmarks."""

import time
import numpy as np
import pandas as pd

from config import Params
from utils.logger import Logger


class ReinforcementLearningNAS:
    """Policy-gradient NAS controller leveraging per-layer logits."""

    def __init__(self,
                 obj_interface,
                 budget=5000,
                 alpha=0.15,
                 baseline_beta=0.1,
                 entropy_beta=0.01,
                 temperature=1.0):
        """
        Args:
            obj_interface: Objective interface (NASInterface) instance.
            budget (int): Number of architecture evaluations.
            alpha (float): Learning rate for the controller updates.
            baseline_beta (float): Momentum term for running reward baseline.
            entropy_beta (float): Entropy regularization weight.
            temperature (float): Softmax temperature for action sampling.
        """
        self.obj_interface = obj_interface
        self.budget = budget
        self.alpha = alpha
        self.baseline_beta = baseline_beta
        self.entropy_beta = entropy_beta
        self.temperature = max(temperature, 1e-3)

        config = Params.search_space_config()
        self.depth = config['depth']
        self.operations = list(config['operations']['search_space'])

        if not self.operations:
            raise RuntimeError('Search space contains no operations for RL controller.')

        self.logits = np.zeros((self.depth, len(self.operations)))
        self.running_baseline = 0.0
        self.results_df = None

    def _sample_architecture(self):
        """Sample architecture actions using softmax per layer."""
        selected_indices = []
        selected_ops = []
        probs_snapshot = []

        for layer in range(self.depth):
            probs = self._softmax(self.logits[layer])
            op_idx = int(np.random.choice(len(self.operations), p=probs))

            probs_snapshot.append(probs)
            selected_indices.append(op_idx)
            selected_ops.append(self.operations[op_idx])

        arch = self._encode_architecture(selected_ops)
        return arch, selected_indices, probs_snapshot

    def _softmax(self, logits):
        z = (logits / self.temperature) - np.max(logits / self.temperature)
        exp_z = np.exp(z)
        return exp_z / exp_z.sum()

    @staticmethod
    def _encode_architecture(ops):
        """Encode operations list into HiveNAS architecture string."""
        return '|'.join(['input'] + ops + ['output'])

    def _update_preferences(self, indices, probs_snapshot, advantage):
        """Update controller logits using REINFORCE with entropy regularizer."""
        if not np.isfinite(advantage):
            return

        scaled_adv = np.clip(advantage, -1.0, 1.0)

        for layer, op_idx in enumerate(indices):
            probs = probs_snapshot[layer]
            one_hot = np.zeros_like(probs)
            one_hot[op_idx] = 1.0
            grad = one_hot - probs
            entropy_grad = -self.entropy_beta * (np.log(probs + 1e-9) + 1.0)
            self.logits[layer] += self.alpha * (scaled_adv * grad + entropy_grad)

    def optimize(self):
        """Run the RL controller for a fixed evaluation budget."""
        Logger.start_log()

        cols = ['itr', 'candidate', 'fitness', 'final_acc', 'epochs', 'params', 'time']
        self.results_df = pd.DataFrame(columns=cols)

        best_fitness = None
        start_time = time.time()

        try:
            from tqdm import tqdm
            import sys
            pbar = tqdm(range(self.budget),
                        desc="RL Optimization",
                        ncols=100,
                        file=sys.stdout,
                        leave=False)
        except ImportError:
            pbar = range(self.budget)

        for itr in pbar:
            arch, action_indices, probs_snapshot = self._sample_architecture()

            eval_start = time.time()
            result = self.obj_interface.evaluate(arch)
            eval_time = time.time() - eval_start

            if isinstance(result, dict):
                fitness = result['fitness']
                final_acc = result.get('final_acc', fitness)
                epochs = result.get('epochs', 0)
                params = result.get('params', 0)
            else:
                fitness = result
                final_acc = fitness
                epochs = 0
                params = 0

            if best_fitness is None:
                best_fitness = fitness
            else:
                if not self.obj_interface.is_minimize and fitness > best_fitness:
                    best_fitness = fitness
                elif self.obj_interface.is_minimize and fitness < best_fitness:
                    best_fitness = fitness

            # Update running baseline then preferences
            self.running_baseline = ((1 - self.baseline_beta) * self.running_baseline
                                     + self.baseline_beta * fitness)
            advantage = fitness - self.running_baseline
            self._update_preferences(action_indices, probs_snapshot, advantage)

            row = pd.DataFrame([{
                'itr': itr,
                'candidate': arch,
                'fitness': fitness,
                'final_acc': final_acc,
                'epochs': epochs,
                'params': params,
                'time': eval_time
            }])
            self.results_df = pd.concat([self.results_df, row], ignore_index=True)

            if hasattr(pbar, 'set_postfix'):
                pbar.set_postfix({
                    'best_fitness': f'{best_fitness:.4f}',
                    'current': f'{fitness:.4f}'
                })

            if (itr + 1) % max(1, (self.budget // 10)) == 0:
                writer = pbar if hasattr(pbar, 'write') else None
                Logger.status(
                    itr,
                    f'Best fitness: {best_fitness}, Total time (s): {time.time() - start_time}',
                    writer=writer
                )

        Logger.end_log()
        return self.results_df
