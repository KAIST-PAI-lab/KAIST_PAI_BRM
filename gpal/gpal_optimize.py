from sklearn.gaussian_process import GaussianProcessRegressor
import numpy as np
from functools import reduce
from typing import Callable
import numpy.typing as npt
from gpal.gpr_fit_predict import *
#from gpal.gpr_instance import GPRInstance
from gpal.utils import *


## args
## Nf: The batch size (length in 0-th dimension) of data for fitting
## Np: The batch size (length in 0-th diemnsion) of data for prediction
## kti: KernelTypeIndex of argsConstructor
## pl: paramsList of argsConstructor
## mi: mulIdx of GPRInstance
## si: sumIdx of GPRInstance
## alpha: alpha of GPRInstance
## n_y: normalize_y of GPRInstance
## n_r_o: n_restarts_optimizer of GPRInstance
## r_s: random_state of GPRInstance


def gpal_optimize2D(gpr:GaussianProcessRegressor, dv1:npt.NDArray[np.float64], dv2:npt.NDArray[np.float64], 
                  est:npt.NDArray[np.float64], dv1Spec:list, dv2Spec:list, 
                  masking:Callable, r_s: bool, r_c:bool):
    if not isinstance(gpr, GaussianProcessRegressor):
        raise TypeError(f"gpr should be a GaussianProcessRegressor instance, got {type(gpr).__name__}.")
    if not isinstance(dv1, np.ndarray):
        raise TypeError(f"dv1 should be a numpy array.")
    if dv1.dtype!=np.float64:
        raise TypeError(f"dv1 should have a dtype of np.float64.")
    if dv1.ndim!=1:
        raise ValueError(f"dv1 should be a 1D array.")
    N=dv1.shape[0]
    if not isinstance(dv2, np.ndarray):
        raise TypeError(f"dv2 should be a numpy array.")
    if dv2.dtype!=np.float64:
        raise TypeError(f"dv2 should have a dtype of np.float64.")
    if dv2.ndim!=1:
        raise ValueError(f"dv2 should be a 1D array.")
    if dv2.shape[0]!=N:
        raise ValueError(f"dv2 should be of length {N}, got {dv2.shape[0]}.")
    if not isinstance(est, np.ndarray):
        raise TypeError(f"est should be a numpy array.")
    if est.dtype!=np.float64:
        raise TypeError(f"est should have a dtype of np.float64.")
    if est.ndim!=1:
        raise ValueError(f"ests should be a 1D array.")
    if est.shape[0]!=N:
        raise ValueError(f"ests should be of length {N}, got {est.shape[0]}.")
    if not isinstance(dv1Spec, list):
        raise TypeError(f"dv1Spec should be a list.")
    if len(dv1Spec)!=3:
        raise ValueError(f"dv1Spec should have a length of 3, got {len(dv1Spec)}.")
    if not isinstance(dv1Spec[2], int):
        raise TypeError(f"The last element of dv1Spec should be an integer value.")
    if not isinstance(dv2Spec, list):
        raise TypeError(f"dv2Spec should be a list.")
    if len(dv2Spec)!=3:
        raise ValueError(f"dv2Spec should have a length of 3, got {len(dv2Spec)}.")
    if not isinstance(dv2Spec[2], int):
        raise TypeError(f"The last element of dv2Spec should be an integer value.")
    if not callable(masking):
        raise TypeError(f"masking should be a callable function.")
    if not isinstance(r_s, bool):
        raise TypeError(f"r_s should be a bool value, got {type(r_s).__name__}.")
    if not isinstance(r_c, bool):
        raise TypeError(f"r_c should be a bool value, got {type(r_c).__name__}.")    
    
    X_fit=np.column_stack([dv1, dv2])
    y_fit=est

    dv1_cands=np.linspace(dv1Spec[0], dv1Spec[1], dv1Spec[2])
    dv2_cands=np.linspace(dv2Spec[0], dv2Spec[1], dv2Spec[2])

    dv1_grid, dv2_grid=np.meshgrid(dv1_cands, dv2_cands, indexing='ij')
    dv1_grid=dv1_grid.ravel()
    dv2_grid=dv2_grid.ravel()

    mask=masking(dv1_grid, dv2_grid)  # Masking the 2D coordinates with a lambda equation
    dvs_grid=np.column_stack([dv1_grid, dv2_grid])
    X_pred=dvs_grid[mask,:]
    lml=gprFit2D(gpr, X_fit, y_fit)
    pred=gprPredict2D(gpr, X_pred, r_s, r_c)
    nextX, pMean, pStd=nextDesign2D(pred, X_pred)


    return nextX, pMean, pStd, lml









def gpal_optimize1D(gpr:GaussianProcessRegressor, dv1:npt.NDArray[np.float64], 
                  est:npt.NDArray[np.float64], dv1Spec:list,
                  masking:Callable, r_s: bool, r_c:bool):
    if not isinstance(gpr, GaussianProcessRegressor):
        raise TypeError(f"gpr should be a GaussianProcessRegressor instance, got {type(gpr).__name__}.")
    if not isinstance(dv1, np.ndarray):
        raise TypeError(f"dv1 should be a numpy array.")
    if dv1.dtype!=np.float64:
        raise TypeError(f"dv1 should have a dtype of np.float64.")
    if dv1.ndim!=2 or dv1.shape[1]!=1:
        raise ValueError(f"dv1 should be a 2D numpy array with a single column, got the shape of {dv1.shape}.")
    N=dv1.shape[0]
    if not isinstance(est, np.ndarray):
        raise TypeError(f"est should be a numpy array.")
    if est.dtype!=np.float64:
        raise TypeError(f"est should have a dtype of np.float64.")
    if est.ndim!=1:
        raise ValueError(f"ests should be a 1D array.")
    if est.shape[0]!=N:
        raise ValueError(f"ests should be of length {N}, got {est.shape[0]}.")
    if not isinstance(dv1Spec, list):
        raise TypeError(f"dv1Spec should be a list.")
    if len(dv1Spec)!=3:
        raise ValueError(f"dv1Spec should have a length of 3, got {len(dv1Spec)}.")
    if not isinstance(dv1Spec[2], int):
        raise TypeError(f"The last element of dv1Spec should be an integer value.")
    if not callable(masking):
        raise TypeError(f"masking should be a callable function.")
    if not isinstance(r_s, bool):
        raise TypeError(f"r_s should be a bool value, got {type(r_s).__name__}.")
    if not isinstance(r_c, bool):
        raise TypeError(f"r_c should be a bool value, got {type(r_c).__name__}.")    
    
    X_fit=dv1
    y_fit=est

    dv1_grid=np.linspace(dv1Spec[0], dv1Spec[1], dv1Spec[2])

    mask=masking(dv1_grid)  # Masking the 1D coordinates with a lambda equation
    X_pred=np.expand_dims(dv1_grid[mask], -1)

    lml=gprFit1D(gpr, X_fit, y_fit)
    pred=gprPredict1D(gpr, X_pred, r_s, r_c)
    nextX, pMean, pStd=nextDesign1D(pred, X_pred)


    return nextX, pMean, pStd, lml




