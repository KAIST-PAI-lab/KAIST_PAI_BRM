from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *
import numpy as np
from functools import reduce
from typing import Optional
import numpy.typing as npt
from nlt.gpal.gpr_instance import KernelInstance
from nlt.gpal.gpr_instance import GPRInstance
from nlt.gpal.utils import prediction

def gpal_optimize():
    KernelInstance