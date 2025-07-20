from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *
import numpy as np
from functools import reduce
from typing import Optional
import numpy.typing as npt
from nlt.gpal.gpr_instance import KernelInstance
from nlt.gpal.gpr_instance import GPRInstance
from nlt.gpal.utils import prediction

def gprFit(gpr:GaussianProcessRegressor, fitData: npt.NDArray[np.float64], obsData: npt.NDArray[np.float64]):
    N=fitData.shape[0]
    assert fitData.shape==(N,1), "The fitting data should be a 2-D numpy array, with the last dimension equal to 1."
    assert obsData.shape==(N,), "The observation data should be a 1-D numpy array."
    gpr.fit(fitData, obsData)
    lml=gpr.log_marginal_likelihood()
    return gpr, lml

def gprPredict(gpr:GaussianProcessRegressor, inputData: npt.NDArray[np.float64], return_std:bool=False, return_cov:bool=False):
    assert not (return_std and return_cov), "At most one of return_std and return_cov can be True."
    N=inputData.shape[0]
    assert inputData.shape==(N,1), "The input data for predictions hould be a 2-D numpy array, with the last dimension equal to 1."
    mean=None
    std=None
    cov=None
    if return_std and not return_cov:
        mean, std=gpr.predict(inputData, return_std, return_cov)
    if not return_std and return_cov:
        mean, cov=gpr.predict(inputData, return_std, return_cov)
    if not return_std and not return_cov:
        mean = gpr.predict(inputData, return_std, return_cov)
    
    preds=prediction(mean=mean, std=std, cov=cov)
    return prediction

def next_design(preds:prediction, inputData:npt.NDArray[np.float64]):
    predMean=prediction.mean
    predStd=prediction.std
    predCov=prediction.cov

    assert predMean is not None, "predMean should be a valid numpy array."
    assert predStd is not None, "predStd should be a valid numpy array."

    max_std_idx=np.argwhere(predStd==np.max(predStd))
    next_design_idx=max_std_idx[np.random.choice(len(max_std_idx))]
    next_design=np.squeeze(inputData[next_design_idx])
    
    return next_design