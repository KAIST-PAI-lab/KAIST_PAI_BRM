from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *
import numpy as np
from functools import reduce
from typing import Optional
import numpy.typing as npt
from nlt.gpal.utils import prediction

def gprFit(gpr:GaussianProcessRegressor, fitData: npt.NDArray[np.float64], obsData: npt.NDArray[np.float64]):
    if not isinstance(gpr, GaussianProcessRegressor):
        raise TypeError(f"gpr should be a GaussianProcessRegressor instance, got {type(gpr).__name__}.")
    N=fitData.shape[0]
    if not isinstance(fitData, np.ndarray):
        raise TypeError(f"fitData should be a numpy array.")
    if not all([isinstance(fd, np.floating) for fd in fitData]):
        idx=[isinstance(fd, str) for fd in fitData].index(False)
        typ=fitData[idx]
        raise TypeError(f"fitData should contain float values, got {type(typ).__name__} at {idx}-th index.")
    if fitData.shape!=(N,2):
         raise ValueError(f"fitData should be a 2-D numpy array with two columns, got {fitData.shape[1]} columns.")
    if not isinstance(obsData, np.ndarray):
        raise TypeError(f"obsData should be a numpy array.")
    if not all([isinstance(od, np.floating) for od in obsData]):
        idx=[isinstance(od, str) for od in obsData].index(False)
        typ=obsData[idx]
        raise TypeError(f"obsData should contain float values, got {type(typ).__name__} at {idx}-th index.")
    if obsData.shape!=(N,):
         raise ValueError(f"obsData should be a 1-D numpy array, got the shape of {fitData.shape}.")    

    gpr.fit(fitData, obsData)
    lml=gpr.log_marginal_likelihood()
    return lml

def gprPredict(gpr:GaussianProcessRegressor, inputData: npt.NDArray[np.float64], returnStd:bool=False, returnCov:bool=False):
    if not isinstance(gpr, GaussianProcessRegressor):
        raise TypeError(f"gpr should be a GaussianProcessRegressor instance, got {type(gpr).__name__}.")
    if returnStd and returnCov:
        raise ValueError(f"At most one of returnStd and returnCov can be True.")
    if not isinstance(inputData, np.ndarray):
        raise TypeError(f"inputData should be a numpy array.")
    if not all([isinstance(id, np.floating) for id in inputData]):
        idx=[isinstance(id, str) for id in inputData].index(False)
        typ=inputData[idx]
        raise TypeError(f"inputData should contain float values, got {type(typ).__name__} at {idx}-th index.")
    if inputData.shape[1]!=2:
         raise ValueError(f"inputData should be a 2-D numpy array with two columns, got {inputData.shape[1]} columns.")
    
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
    return preds

def nextDesign(preds:prediction, inputData:npt.NDArray[np.float64]):
    if not isinstance(preds, prediction):
        raise TypeError(f"preds should be a value returned from gprPredict() function.")
    if not isinstance(inputData, np.ndarray):
        raise TypeError(f"inputData should be a numpy array.")
    if not all([isinstance(id, np.floating) for id in inputData]):
        idx=[isinstance(id, str) for id in inputData].index(False)
        typ=inputData[idx]
        raise TypeError(f"inputData should contain float values, got {type(typ).__name__} at {idx}-th index.")
    if inputData.shape[1]!=2:
        raise ValueError(f"inputData should be a 2-D numpy array with two columns, got {inputData.shape[1]} columns.")    
    if prediction.mean is None:
        raise ValueError(f"The value of 'mean' field of preds should not be None.")
    if prediction.std is None:
        raise ValueError(f"The value of 'std' field of preds should not be None.") 
    predMean=prediction.mean
    predStd=prediction.std
    predCov=prediction.cov

    maxStdIdx=np.argwhere(predStd==np.max(predStd))
    nextDesignIdx=maxStdIdx[np.random.choice(len(maxStdIdx))]
    nextDesign=np.squeeze(inputData[nextDesignIdx])
    
    return nextDesign, predMean, predStd