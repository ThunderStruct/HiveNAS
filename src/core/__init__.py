"""Contains all ABC and NAS-related \
subpackages and modules.

The Artificial Bee Colony package is designed to operate over any optimization task, 
not just NAS, given that the task conforms to the 
:class:`~core.objective_interface.ObjectiveInterface` hooks.
"""

from .nas import NASInterface
from .abc import ArtificialBeeColony

