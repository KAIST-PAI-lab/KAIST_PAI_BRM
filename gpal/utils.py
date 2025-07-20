from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *
import numpy as np
from enum import Enum
from typing import NamedTuple
from typing import Optional
from typing import Union
from typing import Tuple

BoundsType=Union[Tuple[float, float], str]

class enumK(Enum):
    CONSTANT = 1
    DOT = 2
    EXPSINE = 3
    EXPON = 4
    BASE = 5
    MATERN = 6
    PAIRWISE = 7
    PRODUCT = 8
    RBF = 9
    RATQUAD = 10
    SUM = 11
    WHITE = 12

class prediction(NamedTuple):
    mean: np.ndarray
    std: Optional[np.ndarray] = None
    cov: Optional[np.ndarray] = None

def boundsCheck(arg:BoundsType, tupleType=float, boundMin=0.0, allowed=["fixed"]):
    if isinstance(arg, tuple):
        if len(arg)!=2:
            raise ValueError("The bounds tuple should be of length 2.")
        if not (isinstance(arg[0], tupleType) and isinstance(arg[1], tupleType)):
            raise TypeError(f"The bounds values should be of type {tupleType}.")
        if not (arg[0]>=boundMin and arg[1]>=boundMin):
            raise ValueError(f"The bounds values should be larger or equal to {boundMin}")
        
    elif isinstance(arg, str):
        if arg not in allowed:
            raise ValueError(f"The allowed string values for arg2 are {allowed}.")
        
    else:
        raise TypeError("arg2 should be either a tuple of float values or a string.")
        
def constantKernelArgs(arg1:float=1.0, arg2:BoundsType=(1e-5, 1e5)):
    if not isinstance(arg1, float):
        raise TypeError("arg1 should be a float value.")

    boundsCheck(arg2)

    return {'constant_value':arg1, 'constant_value_bounds':arg2}


def dotproductKernelArgs(arg1:float=1.0, arg2:BoundsType=(1e-5, 1e5)):
    if not isinstance(arg1, float):
        raise TypeError("arg1 should be a float value.")
    if not arg1>=0:
        raise ValueError("The sigma_0 parameter for DotProduct kernel should be non-negative.")

    boundsCheck(arg2)
    
    return {'sigma_0':arg1, 'sigma_0_bounds':arg2}


def expsinesquaredKernelArgs(arg1:float=1.0, arg2:float=1.0, arg3:BoundsType=(1e-5, 1e5), arg4:BoundsType=(1e-5, 1e5)):
    if not isinstance(arg1, float):
        raise TypeError("arg1 should be a float value.")
    if not arg1>0:
        raise ValueError("arg1 should be positive.")
    
    if not isinstance(arg2, float):
        raise TypeError("arg2 should be a float value.")
    if not arg2>0:
        raise ValueError("arg2 should be positive.")
    
    boundsCheck(arg3)
    boundsCheck(arg4)

    return {'length_scale':arg1, 'periodicity':arg2, 'length_scale_bounds':arg3, 'periodicity_bounds':arg4}


def exponentiationKernelArgs(arg1:Kernel, arg2:float):
    if not isinstance(arg1, Kernel):
        raise TypeError("arg1 should be a valid kernel instance.")
    if not isinstance(arg2, float):
        raise TypeError("arg2 should be a float value.")
    
    return {'kernel':arg1, 'exponent':arg2}


def maternKernelArgs()



def argsConstructor(N:int, kernelTypeIdx:list[int], paramsList:list[dict]):
    kernelTypeDic={0:'ConstantKernel', 
                   1:'DotProduct', 
                   2:'ExpSineSquared', 
                   3:'Exponentiation',
                   4:'Matern', 
                   5:'PairwiseKernel', 
                   6:'RBF', 
                   7:'RationalQuadratic', 
                   8:'WhiteKernel'}
