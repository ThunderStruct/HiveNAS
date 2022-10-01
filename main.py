import os
import plaidml.keras
from config import Params
from utils import Logger
from benchmarks import Sphere, Rosenbrock
from core import HiveNAS, ArtificialBeeColony


plaidml.keras.install_backend()
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

''' Run HiveNAS '''

Logger.EVALUATION_LOGGING = True

# Objective function selector
if Params['OPTIMIZATION_OBJECTIVE'] == 'HiveNAS':
    objective_interface = HiveNAS()     
elif Params['OPTIMIZATION_OBJECTIVE'] == 'Sphere_min':
    objective_interface = Sphere(10)
elif Params['OPTIMIZATION_OBJECTIVE'] == 'Sphere_max':
    objective_interface = Sphere(10, False)
elif Params['OPTIMIZATION_OBJECTIVE'] == 'Rosenbrock':
    objective_interface = Rosenbrock(2)

abc = ArtificialBeeColony(objective_interface)

# Main optimization loop
abc.optimize()