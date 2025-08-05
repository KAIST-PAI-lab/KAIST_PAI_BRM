from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *
import numpy as np
from functools import reduce
from typing import Optional
import numpy.typing as npt
from gpal.utils import prediction

def gprFit2D(gpr:GaussianProcessRegressor, fitData: npt.NDArray[np.float64], obsData: npt.NDArray[np.float64]):
    if not isinstance(gpr, GaussianProcessRegressor):
        raise TypeError(f"gpr should be a GaussianProcessRegressor instance, got {type(gpr).__name__}.")
    if not isinstance(fitData, np.ndarray):
        raise TypeError(f"fitData should be a numpy array.")
    if fitData.dtype!=np.float64:
        raise TypeError(f"fitData should have a dtype of np.float64.")
    N=fitData.shape[0]
    if fitData.shape!=(N,2):
         raise ValueError(f"fitData should be a 2D numpy array with two columns, got {fitData.shape[1]} columns.")
    if not isinstance(obsData, np.ndarray):
        raise TypeError(f"obsData should be a numpy array.")
    if obsData.dtype!=np.float64:
        raise TypeError(f"obsData should have a dtype of np.float64.")
    if obsData.shape!=(N,):
         raise ValueError(f"obsData should be a 1D numpy array, got the shape of {obsData.shape}.")    

    gpr.fit(fitData, obsData)
    lml=gpr.log_marginal_likelihood()
    return lml


def gprPredict2D(gpr:GaussianProcessRegressor, inputData: npt.NDArray[np.float64], returnStd:bool=False, returnCov:bool=False):
    if not isinstance(gpr, GaussianProcessRegressor):
        raise TypeError(f"gpr should be a GaussianProcessRegressor instance, got {type(gpr).__name__}.")
    if returnStd and returnCov:
        raise ValueError(f"At most one of returnStd and returnCov can be True.")
    if not isinstance(inputData, np.ndarray):
        raise TypeError(f"inputData should be a numpy array.")
    if inputData.dtype!=np.float64:
        raise TypeError(f"inputData should have a dtype of np.float64.")
    if inputData.shape[1]!=2:
         raise ValueError(f"inputData should be a 2D numpy array with two columns, got {inputData.shape[1]} columns.")
    
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


def nextDesign2D(preds:prediction, inputData:npt.NDArray[np.float64]):
    if not isinstance(preds, prediction):
        raise TypeError(f"preds should be a value returned from gprPredict() function.")
    if not isinstance(inputData, np.ndarray):
        raise TypeError(f"inputData should be a numpy array.")
    if inputData.dtype!=np.float64:
        raise TypeError(f"inputData should have a dtype of np.float64.")
    if inputData.shape[1]!=2:
        raise ValueError(f"inputData should be a 2D numpy array with two columns, got {inputData.shape[1]} columns.")    
    if preds.mean is None:
        raise ValueError(f"The value of 'mean' field of preds should not be None.")
    predMean=preds.mean
    if preds.std is not None:
        predStd=preds.std
    if preds.cov is not None:
        predCov=preds.cov

    maxStdIdx=np.argmax(predStd)
    nextDesign=inputData[maxStdIdx]
    
    return nextDesign, predMean[maxStdIdx], predStd[maxStdIdx]






def gprFit1D(gpr:GaussianProcessRegressor, fitData: npt.NDArray[np.float64], obsData: npt.NDArray[np.float64]):
    if not isinstance(gpr, GaussianProcessRegressor):
        raise TypeError(f"gpr should be a GaussianProcessRegressor instance, got {type(gpr).__name__}.")
    if not isinstance(fitData, np.ndarray):
        raise TypeError(f"fitData should be a numpy array.")
    if fitData.dtype!=np.float64:
        raise TypeError(f"fitData should have a dtype of np.float64.")
    N=fitData.shape[0]
    if fitData.shape!=(N,1):
        raise ValueError(f"fitData should be a 2D numpy array with a single column, got the shape of {fitData.shape}.")
    if not isinstance(obsData, np.ndarray):
        raise TypeError(f"obsData should be a numpy array.")
    if obsData.dtype!=np.float64:
        raise TypeError(f"obsData should have a dtype of np.float64.")
    if obsData.shape!=(N,):
         raise ValueError(f"obsData should be a 1D numpy array, got the shape of {obsData.shape}.")    

    gpr.fit(fitData, obsData)
    lml=gpr.log_marginal_likelihood()
    return lml


def gprPredict1D(gpr:GaussianProcessRegressor, inputData: npt.NDArray[np.float64], returnStd:bool=False, returnCov:bool=False):
    if not isinstance(gpr, GaussianProcessRegressor):
        raise TypeError(f"gpr should be a GaussianProcessRegressor instance, got {type(gpr).__name__}.")
    if returnStd and returnCov:
        raise ValueError(f"At most one of returnStd and returnCov can be True.")
    if not isinstance(inputData, np.ndarray):
        raise TypeError(f"inputData should be a numpy array.")
    if inputData.dtype!=np.float64:
        raise TypeError(f"inputData should have a dtype of np.float64.")
    if inputData.ndim!=2 or inputData.shape!=(inputData.shape[0],1):
         raise ValueError(f"inputData should be a 2D numpy array with a single column, got the shape of {inputData.shape}")
    
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


def nextDesign1D(preds:prediction, inputData:npt.NDArray[np.float64]):
    if not isinstance(preds, prediction):
        raise TypeError(f"preds should be a value returned from gprPredict() function.")
    if not isinstance(inputData, np.ndarray):
        raise TypeError(f"inputData should be a numpy array.")
    if inputData.dtype!=np.float64:
        raise TypeError(f"inputData should have a dtype of np.float64.")
    if inputData.ndim!=2 or inputData.shape[1]!=1:
        raise ValueError(f"inputData should be a 2D numpy array with a single column, got the shape of {inputData.shape[1]}.")    
    if preds.mean is None:
        raise ValueError(f"The value of the field 'mean' of preds should not be None.")
    predMean=preds.mean
    if preds.std is not None:
        predStd=preds.std
    if preds.cov is not None:
        predCov=preds.cov

    maxStdIdx=np.argmax(predStd)
    nextDesign=inputData[maxStdIdx]
    
    return nextDesign, predMean[maxStdIdx], predStd[maxStdIdx]