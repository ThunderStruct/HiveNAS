"""Modules relating to numerical benchmarks. \
Although **HiveNAS** is built as a NAS framework, \
numerical benchmarks are used to study the \
optimization process and help empirically \
deduce the best optimizer configuration.
"""

from .rosenbrock import Rosenbrock
from .sphere import Sphere