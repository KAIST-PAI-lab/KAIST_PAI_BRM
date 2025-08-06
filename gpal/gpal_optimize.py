from sklearn.gaussian_process import GaussianProcessRegressor
import numpy as np
from functools import reduce
from typing import Callable
import numpy.typing as npt
import inspect
from gpal.gpr_fit_predict import *
#from gpal.gpr_instance import GPRInstance
from gpal.utils import *



def gpal_optimize(gpr:GaussianProcessRegressor, nDV: int, dvs:npt.NDArray[np.float64], 
                  est:npt.NDArray[np.float64], dvSpecs:list[list], 
                  masking:Callable, r_s: bool = True, r_c:bool = False):
    if not isinstance(gpr, GaussianProcessRegressor):
        raise TypeError(f"gpr should be a GaussianProcessRegressor instance, got {type(gpr).__name__}.")
    if not isinstance(dvs, np.ndarray):
        raise TypeError(f"dvs should be a numpy array.")
    if not isinstance(nDV, int):
        raise TypeError(f"nDV should be an integer value.")
    if nDV<1:
        raise ValueError(f"nDV should be a positive integer.")
    if dvs.dtype!=np.float64:
        raise TypeError(f"dvs should have a dtype of np.float64.")
    if dvs.ndim!=2:
        raise ValueError(f"dvs should be a 2D array, got the shape of {dvs.shape}.")
    if dvs.shape[0]!=nDV:
        raise ValueError(f"dvs should have {nDV} rows; got {dvs.shape[0]} rows.")
    N=dvs.shape[1]
    if not isinstance(est, np.ndarray):
        raise TypeError(f"est should be a numpy array.")
    if est.dtype!=np.float64:
        raise TypeError(f"est should have a dtype of np.float64.")
    if est.ndim!=1:
        raise ValueError(f"ests should be a 1D array.")
    if est.shape[0]!=N:
        raise ValueError(f"ests should be of length {N}, got {est.shape[0]}.")
    if not isinstance(dvSpecs, list):
        raise TypeError(f"dvSpecs should be a list.")
    if not all([isinstance(spec, list) for spec in dvSpecs]):
        raise TypeError(f"dvSpecs should contain list elements.")
    if len(dvSpecs)!=nDV:
        raise ValueError(f"dvSpecs should have {nDV} list elements, got {len(dvSpecs)}.")
    if not all([isinstance(spec[2], int) for spec in dvSpecs]):
        raise TypeError(f"The last element of lists in dvSpecs should be an integer value.")
    if not callable(masking):
        raise TypeError(f"masking should be a callable function.")
    numP=len(inspect.signature(masking).parameters)
    if numP != nDV:
        raise ValueError(f"masking should have {nDV} parameters, got {numP} parameters")
    if not isinstance(r_s, bool):
        raise TypeError(f"r_s should be a bool value, got {type(r_s).__name__}.")
    if not isinstance(r_c, bool):
        raise TypeError(f"r_c should be a bool value, got {type(r_c).__name__}.")    
    
    X_fit=dvs.T
    y_fit=est

    dv_cands=[]
    for i in range(nDV):
        dv_cands.append(np.linspace(dvSpecs[i][0], dvSpecs[i][1], dvSpecs[i][2]))

    grids=list(np.meshgrid(*dv_cands, indexing='ij'))


    coords=np.stack(grids,-1).T
    mask=masking(*coords)  # Masking the 2D coordinates with a lambda equation
    X_pred=coords[:,mask].T
    lml=gprFit(gpr, nDV, X_fit, y_fit)
    pred=gprPredict(gpr, nDV, X_pred, r_s, r_c)
    nextX, pMean, pStd=nextDesign(pred, nDV, X_pred)


    return nextX, pMean, pStd, lml




def gpal_optimize2D(gpr:GaussianProcessRegressor, dv1:npt.NDArray[np.float64], dv2:npt.NDArray[np.float64], 
                  est:npt.NDArray[np.float64], dv1Spec:list, dv2Spec:list, 
                  masking:Callable, r_s: bool = True, r_c:bool = False):
    if not isinstance(gpr, GaussianProcessRegressor):
        raise TypeError(f"gpr should be a GaussianProcessRegressor instance, got {type(gpr).__name__}.")
    if not isinstance(dv1, np.ndarray):
        raise TypeError(f"dv1 should be a numpy array.")
    if dv1.dtype!=np.float64:
        raise TypeError(f"dv1 should have a dtype of np.float64.")
    if dv1.ndim!=2 or dv1.shape[1]!=1:
        raise ValueError(f"dv1 should be a 2D array with a single column, got the shape of {dv1.shape}.")
    N=dv1.shape[0]
    if not isinstance(dv2, np.ndarray):
        raise TypeError(f"dv2 should be a numpy array.")
    if dv2.dtype!=np.float64:
        raise TypeError(f"dv2 should have a dtype of np.float64.")
    if dv2.ndim!=2 or dv2.shape[1]!=1:
        raise ValueError(f"dv2 should be a 2D array with a single column, got the shape fo {dv2.shape}.")
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
                  masking:Callable, r_s: bool = True, r_c:bool = False):
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




