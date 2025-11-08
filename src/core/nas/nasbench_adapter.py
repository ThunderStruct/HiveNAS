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
                'epochs': 108,
                'filename': self._get_hash(arch),
                'params': 0
            }

        # Query NASBench-101
        try:
            result = self.api.query(
                arch=arch_spec,
                dataset='cifar10',
                split='test',  # Use test accuracy as fitness
                budget=108     # Use final epoch (108)
            )

            # NASBench API returns tuple: (info_dict, metrics_dict)
            if isinstance(result, tuple) and len(result) == 2:
                info, metrics = result
            else:
                # Unexpected format
                return {
                    'fitness': 0.1,
                    'epochs': 108,
                    'filename': self._get_hash(arch),
                    'params': 0
                }

            # Extract test accuracy from metrics at budget=108
            # metrics[108] is a list of runs (typically 3 runs)
            # We take the mean of final_test_accuracy across runs
            if 108 in metrics and len(metrics[108]) > 0:
                test_accs = [run['final_test_accuracy'] for run in metrics[108]]
                fitness = float(np.mean(test_accs))
            else:
                # Architecture not in benchmark at this budget
                fitness = 0.1

            # Get trainable parameters from info dict
            params = int(info.get('trainable_parameters', 0))

            return {
                'fitness': fitness,
                'epochs': 108,
                'filename': self._get_hash(arch),
                'params': params
            }

        except Exception as e:
            print(f"Warning: Error querying NASBench-101: {e}")
            # Return low fitness on error
            return {
                'fitness': 0.1,
                'epochs': 108,
                'filename': self._get_hash(arch),
                'params': 0
            }

    def _convert_to_nasbench_format(self, arch: list) -> Optional[Any]:
        """
        Convert SwarmNAS architecture to NASBench-101 format.

        NASBench-101 uses:
        - 7x7 adjacency matrix
        - 7 operations: ['input', 'conv3x3-bn-relu', 'conv1x1-bn-relu',
                         'maxpool3x3', 'output']

        This is a simplified mapping. For full compatibility, consider:
        1. Mapping SwarmNAS ops to NASBench ops
        2. Constructing valid adjacency matrix
        3. Handling skip connections

        Args:
            arch (list): SwarmNAS architecture

        Returns:
            Arch101 or None: NASBench-101 architecture spec, or None if incompatible
        """
        from nasbenchapi import Arch101

        # Simplified mapping of SwarmNAS operations to NASBench-101
        # This is a basic approximation - you may want to refine this
        op_mapping = {
            'conv3x3_64bnreluavgpool': 'conv3x3-bn-relu',
            'conv3x3_128bnreluavgpool': 'conv3x3-bn-relu',
            'conv3x3_256bnreluavgpool': 'conv3x3-bn-relu',
            'conv3x3_16bnrelu': 'conv3x3-bn-relu',
            'conv3x3_32bnrelu': 'conv3x3-bn-relu',
            'resx2reg_32_conv3x3_64bnrelu': 'conv3x3-bn-relu',
            'resx1reg_128_conv3x3_256bnrelu': 'conv3x3-bn-relu',
            'resx1reg_128_conv3x3_128bnrelu': 'conv3x3-bn-relu',
            'max_pool3x3': 'maxpool3x3',
            'avg_pool3x3': 'maxpool3x3',
        }

        # Limit to 5 operations (input + 3 middle + output = 5, expandable to 7)
        max_ops = min(len(arch), 5)

        # Map operations
        operations = ['input']
        for op in arch[:max_ops-2]:  # Reserve space for output
            if op.startswith('sc_'):
                # Skip connection - use conv1x1 as approximation
                operations.append('conv1x1-bn-relu')
            else:
                mapped_op = op_mapping.get(op, 'conv3x3-bn-relu')
                operations.append(mapped_op)

        # Pad to 7 operations if needed
        while len(operations) < 6:
            operations.append('conv3x3-bn-relu')
        operations.append('output')

        # Create simple sequential adjacency matrix
        # More sophisticated graph construction could improve compatibility
        adjacency = [[0] * 7 for _ in range(7)]

        # Sequential connections
        for i in range(len(operations) - 1):
            adjacency[i][i + 1] = 1

        # Add some skip connections for richer topology
        if len(operations) >= 5:
            adjacency[0][2] = 1  # Input to third node
            adjacency[1][3] = 1  # Second to fourth node

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
