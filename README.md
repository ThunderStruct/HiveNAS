# SwarmNAS

Neural architecture search with Artificial Bee Colony (ABC), Particle Swarm Optimization (PSO), and Ant Colony Optimization (ACO). Candidates can be evaluated by training networks or querying NASBench-101.

This is the `SwarmNAS` branch. For the ABC framework described in the HiveNAS paper, see [`main`](https://github.com/ThunderStruct/HiveNAS/tree/main).

## Installation

Clone this branch and install its dependencies in a virtual environment. The pinned dependencies target Python 3.9.

```bash
git clone --branch SwarmNAS https://github.com/ThunderStruct/HiveNAS.git
cd HiveNAS
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install -r src/requirements.txt
cd src
```

Run the examples below from `src/`. On Windows, activate the environment with `.venv\Scripts\activate`.

For NASBench-101 evaluation, also install its API and set the dataset path:

```bash
python -m pip install nasbenchapi
export NASBENCH101_PATH=/path/to/nasbench_only108.pkl
```

## Quick Start

Choose the optimizer through its Python class:

```python
from config import Params
from core import NASInterface, ParticleSwarmOptimization

Params.set_parameter('ITERATIONS_COUNT', 20)
Params.set_parameter('USE_NASBENCH', False)

optimizer = ParticleSwarmOptimization(NASInterface(), swarm_size=10)
results = optimizer.optimize()
```

To use ABC or ACO, replace the optimizer construction with one of these:

```python
from core import ArtificialBeeColony, AntColonyOptimization

optimizer = ArtificialBeeColony(NASInterface(), colony_size=10)
# Or:
optimizer = AntColonyOptimization(NASInterface(), colony_size=10)
```

Call `optimizer.optimize()` after constructing the chosen optimizer. To query NASBench-101 instead of training candidates, set `USE_NASBENCH` to `True` before creating `NASInterface()`.

## Configuration

Shared settings, including `ITERATIONS_COUNT`, `USE_NASBENCH`, and `NASBENCH_BUDGET`, are defined in `src/config/params.py`. PSO and ACO settings are constructor arguments:

| Optimizer | Parameters and defaults |
| --- | --- |
| PSO | `swarm_size=10`, `w=0.7`, `c1=1.5`, `c2=1.5` |
| ACO | `colony_size=10`, `alpha=1.0`, `beta=2.0`, `rho=0.1`, `Q=1.0` |

## Command Line

The `HiveNAS.py` command runs ABC. Use the Python classes above for PSO and ACO.

```bash
python HiveNAS.py --verbose --iterations-count 20
python HiveNAS.py --verbose --config-file path/to/config.yaml
```

## Project Structure

```text
HiveNAS/
├── src/
│   ├── HiveNAS.py          # ABC command-line entry point
│   ├── config/             # Shared settings and operation definitions
│   ├── core/
│   │   ├── abc/            # Artificial Bee Colony
│   │   ├── pso/            # Particle Swarm Optimization
│   │   ├── aco/            # Ant Colony Optimization
│   │   └── nas/            # Search space and candidate evaluation
│   └── utils/
├── docs/
├── references.bib
└── CITATION.cff
```

## Documentation

See the [documentation](https://hivenas.readthedocs.io/en/latest/) for HiveNAS usage and the `docs/` directory for this branch's documentation sources.

## License

[MIT License](LICENSE).

## Publications and Citation

For HiveNAS and Artificial Bee Colony architecture search, cite:

“HiveNAS: Neural Architecture Search Using Artificial Bee Colony Optimization.” ACDSA 2026, pp. 1–7. [doi:10.1109/ACDSA67686.2026.11468089](https://doi.org/10.1109/ACDSA67686.2026.11468089).

```bibtex
@inproceedings{shahawy2026hivenas,
  author = {Shahawy, Mohamed},
  title = {{HiveNAS}: Neural Architecture Search Using Artificial Bee Colony Optimization},
  booktitle = {2026 International Conference on Artificial Intelligence, Computer, Data Sciences and Applications (ACDSA)},
  year = {2026},
  pages = {1--7},
  publisher = {IEEE},
  doi = {10.1109/ACDSA67686.2026.11468089},
  url = {https://doi.org/10.1109/ACDSA67686.2026.11468089}
}
```

For SwarmNAS and multi-objective swarm intelligence architecture search, cite:

“Swarm Intelligence for Multi-Objective Neural Architecture Search.” ISMSI 2026, pp. 131–136. [doi:10.1145/3820820.3820826](https://doi.org/10.1145/3820820.3820826).

```bibtex
@inproceedings{shahawy2026swarmnas,
  author = {Shahawy, Mohamed and Sedky, Mohamed},
  title = {Swarm Intelligence for Multi-Objective Neural Architecture Search},
  booktitle = {Proceedings of the 2026 10th International Conference on Intelligent Systems, Metaheuristics \& Swarm Intelligence},
  year = {2026},
  pages = {131--136},
  publisher = {ACM},
  doi = {10.1145/3820820.3820826},
  url = {https://doi.org/10.1145/3820820.3820826}
}
```

If your work uses both HiveNAS and SwarmNAS, cite both publications.
