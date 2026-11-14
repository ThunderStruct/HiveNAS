# HiveNAS

Neural architecture search using Artificial Bee Colony (ABC) optimization, with configurable search spaces and candidate evaluation.

This is the `main` branch. For PSO, ACO, and NASBench-101 support, see [`SwarmNAS`](https://github.com/ThunderStruct/HiveNAS/tree/SwarmNAS).

## Installation

Clone this branch and install its dependencies in a virtual environment. The pinned dependencies target Python 3.9.

```bash
git clone --branch main https://github.com/ThunderStruct/HiveNAS.git
cd HiveNAS
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install -r src/requirements.txt
cd src
```

Run the examples below from `src/`. On Windows, activate the environment with `.venv\Scripts\activate`.

The published HiveNAS package is also available on [PyPI](https://pypi.org/project/hivenas/). The source installation above provides the branch documented here.

## Quick Start

```python
from HiveNAS import HiveNAS
from config import Params

Params.set_parameter('ITERATIONS_COUNT', 20)
HiveNAS.find_topology()
```

## Command Line

```bash
python HiveNAS.py --verbose --iterations-count 20
python HiveNAS.py --verbose --config-file path/to/config.yaml
```

Search spaces, datasets, and evaluation settings are defined in `src/config/params.py` and `src/config/settings/config_default.yaml`.

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
