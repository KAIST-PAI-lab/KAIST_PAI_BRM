from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *
import numpy as np
from functools import reduce
from typing import Optional
import numpy.typing as npt
from nlt.gpal.utils import prediction

def gprFit(gpr:GaussianProcessRegressor, fitData: npt.NDArray[np.float64], obsData: npt.NDArray[np.float64]):
    N=fitData.shape[0]
    assert fitData.shape==(N,1), "The fitting data should be a 2-D numpy array, with the last dimension equal to 1."
    assert obsData.shape==(N,), "The observation data should be a 1-D numpy array."
    gpr.fit(fitData, obsData)
    lml=gpr.log_marginal_likelihood()
    return lml

def gprPredict(gpr:GaussianProcessRegressor, inputData: npt.NDArray[np.float64], returnStd:bool=False, returnCov:bool=False):
    assert not (returnStd and returnCov), "At most one of returnStd and returnCov can be True."
    N=inputData.shape[0]
    assert inputData.shape==(N,1), "The input data for predictions should be a 2-D numpy array, with the last dimension equal to 1."
    mean=None
    std: Optional[np.ndarray] = None
    cov: Optional[np.ndarray] = None
    if returnStd and not returnCov:
        mean, std=gpr.predict(inputData, returnStd, returnCov)
    if not returnStd and returnCov:
        mean, cov=gpr.predict(inputData, returnStd, returnCov)
    if not returnStd and not returnCov:
        mean = gpr.predict(inputData, returnStd, returnCov)
    
    preds=prediction(mean=mean, std=std, cov=cov)
    return prediction

def nextDesign(preds:prediction, inputData:npt.NDArray[np.float64]):
    predMean=prediction.mean
    predStd=prediction.std
    predCov=prediction.cov

    assert predMean is not None, "predMean should be a valid numpy array."
    assert predStd is not None, "predStd should be a valid numpy array."

    maxStdIdx=np.argwhere(predStd==np.max(predStd))
    nextDesignIdx=maxStdIdx[np.random.choice(len(maxStdIdx))]
    nextDesign=np.squeeze(inputData[nextDesignIdx])
    
    return nextDesign