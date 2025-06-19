import os
import sys

sys.path.append(os.path.abspath("/home/filippo/Github/Varational-Monte-Carlo/src/classes"))  # Change the path to a more universal one

import numpy as np
import matplotlib.pyplot as plt
import functools as fct
from functions import Laplacian_op
from Montecarlo import MonteCarlov2

