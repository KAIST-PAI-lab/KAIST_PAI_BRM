from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *
import numpy as np
from functools import reduce
from typing import Optional
import numpy.typing as npt
from nlt.gpal.gpr_fit_predict import gprFit
from nlt.gpal.gpr_fit_predict import gprPredict
from nlt.gpal.gpr_fit_predict import nextDesign
from nlt.gpal.gpr_instance import GPRInstance
from nlt.gpal.utils import *


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


def gpal_optimize(gpr:GaussianProcessRegressor, gns:npt.NDArray[np.float64], ubs:npt.NDArray[np.float64], 
                  gnEsts:npt.NDArray[np.float64], r_s:bool, r_c:bool):
    if not isinstance(gpr, GaussianProcessRegressor):
        raise TypeError(f"gpr should be a GaussianProcessRegressor instance, got {type(gpr).__name__}.")
    if not isinstance(gns, np.ndarray):
        raise TypeError(f"gns should be a 1-D numpy array.")
    if not all(isinstance(gn, np.floating) for gn in gns):
        typ=gns[[isinstance(gn, np.floating) for gn in gns].index(False)]
        raise TypeError(f"gns should contain float values, got {type(typ).__name__ }.")
    if gns.ndim!=1:
        raise ValueError(f"gns should be a 1-D numpy array.")
    N=gns.shape[0]
    if not isinstance(ubs, np.ndarray):
        raise TypeError(f"ubs should be a 1-D numpy array.")
    if not all(isinstance(ub, np.floating) for ub in ubs):
        typ=ubs[[isinstance(ub, np.floating) for ub in ubs].index(False)]
        raise TypeError(f"ubs should contain float values, got {type(typ).__name__ }.")
    if ubs.ndim!=1:
        raise ValueError(f"ubs should be a 1-D numpy array.")
    if ubs.shape[0]!=N:
        raise ValueError(f"ubs should be of length {N}, got {ubs.shape[0]}.")
    if not isinstance(gnEsts, np.ndarray):
        raise TypeError(f"gnEsts should be a 1-D numpy array.")
    if not all(isinstance(gne, np.floating) for gne in gnEsts):
        typ=gnEsts[[isinstance(gne, np.floating) for gne in gnEsts].index(False)]
        raise TypeError(f"gnEsts should contain float values, got {type(typ).__name__ }.")
    if gnEsts.ndim!=1:
        raise ValueError(f"gnEsts should be a 1-D numpy array.")
    if gnEsts.shape[0]!=N:
        raise ValueError(f"gnEsts should be of length {N}, got {gnEsts.shape[0]}.")
    if not isinstance(r_s, bool):
        raise TypeError(f"r_s should be a bool value, got {type(r_s).__name__}.")
    if not isinstance(r_c, bool):
        raise TypeError(f"r_c should be a bool value, got {type(r_c).__name__}.")    
    if not np.all(ubs>gns):
        raise ValueError("Upper bound values should be larger than corresponding given number values.")
    if not np.all(ubs>gnEsts):
        raise ValueError("Upper bound values should be larger than given number estimates at the same index.")
    
    X_fit=np.column_stack([gns, ubs])
    y_fit=gnEsts
    gpr.fit(X_fit, y_fit)

    gn_cands=np.linspace(5, 500, 500-5+1)
    ub_cands=np.linspace(50, 500, 10)

    gn_grid, ub_grid=np.meshgrid(gn_cands, ub_cands, indexing='ij')
    gn_grid=gn_grid.ravel()
    ub_grid=ub_grid.ravel()

    mask=gn_grid<=ub_grid
    X_pred=np.stack([gn_grid[mask], ub_grid[mask]], axis=-1)

    lml=gprFit(gpr, X_fit, y_fit)
    pred=gprPredict(gpr, X_pred, r_s, r_c)
    nextX, pMean, pStd=nextDesign(pred, X_fit)

    next_gn=nextX[0]
    next_ub=nextX[1]

    return next_gn, next_ub, pMean, pStd, lml





