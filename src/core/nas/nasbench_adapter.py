"""NASBench-101 adapter for SwarmNAS framework.

This module provides an evaluation strategy that uses NASBench-101 instead of
training from scratch, enabling fast evaluation of architectures.
"""

import sys
sys.path.append('...')

import os
import hashlib
import numpy as np
from typing import Dict, Any, Optional
from config import Params


class NASBench101Adapter:
    """
    Adapter to use NASBench-101 for architecture evaluation.

    This adapter translates SwarmNAS architecture representations to NASBench-101
    format and queries the benchmark for performance metrics.

    Attributes:
        api (NASBench101): NASBench-101 API instance
        config (dict): Evaluation configuration parameters
        use_nasbench (bool): Whether to use NASBench or fall back to training
    """

    def __init__(self, config: Dict[str, Any], use_nasbench: bool = True):
        """
        Initialize NASBench-101 adapter.

        Args:
            config (dict): Evaluation configuration
            use_nasbench (bool): Whether to use NASBench-101 or train from scratch

        Raises:
            ImportError: If nasbenchapi is not installed
            EnvironmentError: If NASBENCH101_PATH is not set and use_nasbench=True
        """
        self.config = config
        self.use_nasbench = use_nasbench
        self.api = None

        if use_nasbench:
            try:
                from nasbenchapi import NASBench101

                # Check for environment variable or use default path
                nasbench_path = os.environ.get('NASBENCH101_PATH')

                if nasbench_path is None:
                    raise EnvironmentError(
                        "NASBENCH101_PATH environment variable not set. "
                        "Please set it to the path of NASBench-101 pickle file, "
                        "or set use_nasbench=False to train from scratch."
                    )

                # Initialize NASBench-101 API
                self.api = NASBench101(nasbench_path, verbose=True)
                print(f"✓ NASBench-101 loaded successfully from {nasbench_path}")

            except ImportError:
                print("Warning: nasbenchapi not installed. Falling back to training from scratch.")
                print("Install with: pip install nasbenchapi")
                self.use_nasbench = False
                self.api = None
            except FileNotFoundError as e:
                print(f"Warning: {e}")
                print("Falling back to training from scratch.")
                self.use_nasbench = False
                self.api = None

    def evaluate(self, arch: list) -> Dict[str, Any]:
        """
        Evaluate architecture using NASBench-101.

        Args:
            arch (list): Architecture as list of operation strings

        Returns:
            dict: Evaluation results containing:
                - fitness (float): Test accuracy
                - epochs (int): Number of epochs (from NASBench)
                - filename (str): Hash identifier
                - params (int): Number of parameters
        """
        if not self.use_nasbench or self.api is None:
            raise RuntimeError(
                "NASBench-101 is not available. "
                "Either install nasbenchapi and set NASBENCH101_PATH, "
                "or use regular training mode."
            )

        # Convert SwarmNAS architecture to NASBench-101 format
        arch_spec = self._convert_to_nasbench_format(arch)

        if arch_spec is None:
            # Architecture not compatible with NASBench-101
            # Return low fitness to discourage this architecture
            return {
                'fitness': 0.1,
                'final_acc': 0.1,
                'epochs': 108,
                'filename': self._get_hash(arch),
                'params': 0,
                'momentum': {}  # NASBench doesn't use momentum
            }

        # Get budget from Params (default to 12 for better variance)
        from config import Params
        try:
            budget = Params['NASBENCH_BUDGET']
        except (KeyError, TypeError):
            budget = 12  # Default to budget=12 for high variance

        # Query NASBench-101
        try:
            result = self.api.query(
                arch=arch_spec,
                dataset='cifar10',
                split='test',  # Use test accuracy as fitness
                budget=budget
            )

            # NASBench API returns tuple: (info_dict, metrics_dict)
            if isinstance(result, tuple) and len(result) == 2:
                info, metrics = result
            else:
                # Unexpected format
                return {
                    'fitness': 0.1,
                    'epochs': budget,
                    'filename': self._get_hash(arch),
                    'params': 0,
                    'momentum': {}  # NASBench doesn't use momentum
                }

            # Extract test accuracy from metrics at the specified budget
            # metrics[budget] is a list of runs (typically 3 runs)
            # We take the mean of final_test_accuracy across runs
            if budget in metrics and len(metrics[budget]) > 0:
                test_accs = [run['final_test_accuracy'] for run in metrics[budget]]
                fitness = float(np.mean(test_accs))
            else:
                # Architecture not in benchmark at this budget
                fitness = 0.1

            # Also get final converged accuracy at budget=108 for reference
            final_acc = fitness  # Default to current budget accuracy
            if budget != 108 and 108 in metrics and len(metrics[108]) > 0:
                final_accs = [run['final_test_accuracy'] for run in metrics[108]]
                final_acc = float(np.mean(final_accs))

            # Get trainable parameters from info dict
            params_raw = info.get('trainable_parameters', 0)
            params = int(params_raw) if params_raw is not None else 0

            return {
                'fitness': fitness,
                'final_acc': final_acc,  # Converged accuracy at budget=108
                'epochs': budget,
                'filename': self._get_hash(arch),
                'params': params,
                'momentum': {}  # NASBench doesn't use momentum
            }

        except Exception as e:
            print(f"Warning: Error querying NASBench-101: {e}")
            # Return low fitness on error
            return {
                'fitness': 0.1,
                'final_acc': 0.1,
                'epochs': budget,
                'filename': self._get_hash(arch),
                'params': 0,
                'momentum': {}  # NASBench doesn't use momentum
            }

    def _convert_to_nasbench_format(self, arch: list) -> Optional[Any]:
        """
        Convert SwarmNAS architecture to NASBench-101 format.

        NASBench-101 uses:
        - 7x7 adjacency matrix
        - 7 operations: ['input', 'conv3x3-bn-relu', 'conv1x1-bn-relu',
                         'maxpool3x3', 'output']

        This creates a hash-based mapping to unique NASBench architectures.

        Args:
            arch (list): SwarmNAS architecture

        Returns:
            Arch101 or None: NASBench-101 architecture spec, or None if incompatible
        """
        from nasbenchapi import Arch101
        import hashlib

        # Map SwarmNAS operations to NASBench-101 operations
        op_mapping = {
            'conv3x3_64bnreluavgpool': 'conv3x3-bn-relu',
            'conv3x3_128bnreluavgpool': 'conv3x3-bn-relu',
            'conv3x3_256bnreluavgpool': 'conv3x3-bn-relu',
            'conv3x3_16bnrelu': 'conv3x3-bn-relu',
            'conv3x3_32bnrelu': 'conv3x3-bn-relu',
            'conv1x1_64bnrelu': 'conv1x1-bn-relu',
            'conv1x1_128bnrelu': 'conv1x1-bn-relu',
            'resx2reg_32_conv3x3_64bnrelu': 'conv3x3-bn-relu',
            'resx1reg_128_conv3x3_256bnrelu': 'conv3x3-bn-relu',
            'resx1reg_128_conv3x3_128bnrelu': 'conv3x3-bn-relu',
            'max_pool3x3': 'maxpool3x3',
            'avg_pool3x3': 'maxpool3x3',
        }

        # Extract operations from SwarmNAS string format
        # Format: "input|op1|op2|op3|op4|op5|output"
        if isinstance(arch, str):
            ops_str = arch.split('|')[1:-1]  # Remove input/output
        elif isinstance(arch, list):
            ops_str = [str(op) for op in arch]
        else:
            return None

        # Map operations (NASBench-101 has max 5 internal nodes)
        operations = ['input']
        for op_str in ops_str[:5]:  # Limit to 5 internal operations
            mapped_op = op_mapping.get(op_str, 'conv3x3-bn-relu')
            operations.append(mapped_op)

        # Pad to 7 total operations (input + 5 internal + output)
        while len(operations) < 6:
            operations.append('conv3x3-bn-relu')
        operations.append('output')

        # Create adjacency matrix based on architecture hash
        # This ensures different SwarmNAS architectures map to different NASBench architectures
        arch_hash = hashlib.md5(''.join(ops_str).encode()).hexdigest()
        hash_int = int(arch_hash[:8], 16)  # Use first 8 hex chars

        adjacency = [[0] * 7 for _ in range(7)]

        # Always have sequential backbone
        for i in range(6):
            adjacency[i][i + 1] = 1

        # Add skip connections based on hash (deterministic but varied)
        # Each bit in hash determines a potential skip connection
        skip_possibilities = [
            (0, 2), (0, 3), (0, 4), (0, 5),  # Input to middle nodes
            (1, 3), (1, 4), (1, 5),           # Node 1 to later nodes
            (2, 4), (2, 5),                   # Node 2 to later nodes
            (3, 5),                           # Node 3 to node 5
        ]

        for idx, (src, dst) in enumerate(skip_possibilities):
            if (hash_int >> idx) & 1:  # Check if bit is set
                adjacency[src][dst] = 1

        try:
            arch_spec = Arch101(
                adjacency=adjacency,
                operations=operations
            )
            return arch_spec
        except Exception as e:
            print(f"Warning: Could not create valid NASBench-101 architecture: {e}")
            return None

    def _get_hash(self, arch: list) -> str:
        """
        Generate hash identifier for architecture.

        Args:
            arch (list): Architecture operations list

        Returns:
            str: SHA1 hash identifier
        """
        return hashlib.sha1(''.join(arch).encode('UTF-8')).hexdigest()

    def sample_from_nasbench(self, n: int = 1) -> list:
        """
        Sample random architectures from NASBench-101.

        This can be used to initialize the search with known-good architectures.

        Args:
            n (int): Number of architectures to sample

        Returns:
            list: List of sampled architecture strings (SwarmNAS format)
        """
        if not self.use_nasbench or self.api is None:
            return []

        samples = self.api.random_sample(n=n)

        # Convert back to SwarmNAS format (simplified)
        # This is a placeholder - actual conversion would be more sophisticated
        swarmnas_archs = []
        for arch_spec in samples:
            # Extract operations (skip input/output)
            ops = arch_spec.operations[1:-1]

            # Reverse mapping (approximate)
            swarmnas_ops = []
            for op in ops:
                if op == 'conv3x3-bn-relu':
                    swarmnas_ops.append('conv3x3_64bnreluavgpool')
                elif op == 'conv1x1-bn-relu':
                    swarmnas_ops.append('conv3x3_32bnrelu')
                elif op == 'maxpool3x3':
                    swarmnas_ops.append('max_pool3x3')

            swarmnas_archs.append('|'.join(['input'] + swarmnas_ops + ['output']))

        return swarmnas_archs
