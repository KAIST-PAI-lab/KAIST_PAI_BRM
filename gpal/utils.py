from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *
import numpy as np
import numpy.typing as npt
from functools import wraps
from enum import Enum
from typing import NamedTuple
from typing import Optional
from typing import Union
from typing import Tuple
from typing import Callable
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


BoundsType=Union[Tuple[float, float], str]
ScalesType=Union[npt.NDArray[np.float64], float]
npFuncType=Callable[[npt.NDArray, npt.NDArray], npt.NDArray]
MetricsType=Union[str, npFuncType]
kernelTypeDic={0:'ConstantKernel', 
                1:'DotProduct', 
                2:'ExpSineSquared', 
                3:'Exponentiation',
                4:'Matern', 
                5:'PairwiseKernel', 
                6:'RBF', 
                7:'RationalQuadratic', 
                8:'WhiteKernel'}
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
        raise ValueError("arg1 should be non-negative.")

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


def maternKernelArgs(arg1:ScalesType=1.0, arg2:BoundsType=(1e-5, 1e5), arg3:float=1.5):
    if not (isinstance(arg1, np.ndarray) or isinstance(arg1, float)):
        raise TypeError("arg1 should be a 1D numpy array or a float value.")
    if isinstance(arg1, np.ndarray):
        if arg1.ndim!=1:
            raise ValueError("A numpy array arg1 should be a 1-D array.")
        if arg1.dtype!=np.float64:
            raise TypeError("Elements of numpy array arg1 should be of type np.float64, which is a default float type for numpy arrays.")

    boundsCheck(arg2)

    if not isinstance(arg3, float):
        raise TypeError("arg3 should be a float value.")

    return {'length_scale':arg1, 'length_scale_bounds':arg2, 'nu': arg3}            


def pairwiseArg3(arg3:npFuncType):
    @wraps(arg3)
    def checker(data1, data2):
        if not isinstance(data1, np.ndarray):
            raise TypeError(f"The first argument of arg3 callable should be numpy.ndarray")
        if not isinstance(data2, np.ndarray):
            raise TypeError(f"The second argument of arg3 callable should be numpy.ndarray")
        if data1.ndim!=2:
            raise ValueError(f"The first argument of arg3 callable should be a 2-D numpy array, got {data1.shape} as a shape.")
        if data2.ndim!=2:
            raise ValueError(f"The second argument of arg3 callable should be a 2-D numpy array, got {data2.shape} as a shape.")
        if data1.shape[1]!=data2.shape[1]:
            raise ValueError(f"The two arguments of arg3 callable should have the same length in 1-th dimension, got {data1.shape[1]} and {data2.shape[1]}")
        result=arg3(data1, data2)

        if not isinstance(result, np.ndarray):
            raise TypeError(f"The return value of arg3 callable should be numpy.ndarray")
        if result.ndim!=2 or result.shape[0]!=result.shape[1]:
            raise ValueError(f"The return value of arg3 callable should be a 2-D square numpy array, got the shape of{result.shape}.")
        if not np.allclose(result, result.T, atol=1e-10):
            raise ValueError("The return value of arg3 callable should be symmetric.")
        try:
            np.linalg.cholesky(result+1e-10*np.eye(result.shape[0]))
        except np.linalg.LinAlgError:
            raise ValueError("The return value of arg3 callable should be a positive semi-definite kernel matrix.")
        return None
    return checker

## arg4: dict[parameter : argument]
## target: dict[parameter : expected type]
def pairwiseArg4(arg4:dict, target:dict):
    ## Empty dictionary - for additive_chi2 kernel.
    if target=={}:
        return
    paramDict=target
    paramCand=list(paramDict.keys())
    paramCand.sort()
    paramSet=set(paramCand)

    keyList=list(arg4.keys())
    keyIdx=[paramCand.index(k) for k in keyList]
    keySet=set(arg4.keys())
    if not keySet.issubset(paramSet):
        raise ValueError(f"The arg4.keys() should only contain keys included in the following list: {paramCand}. Not all available keys need to be present.")
    for ki in keyIdx:
        poi=keyList[ki]  # Parameter of interest
        aoi=arg4[poi] # Argument of interest
        expType=paramDict[poi] # Expected type of aoi value
        if not isinstance(aoi, paramDict[poi]):
            raise TypeError(f"The arg4[{poi}] should be of type {expType}, got {type(aoi).__name__}")
    return


def pairwisekernelArgs(arg1:float=1.0, arg2:BoundsType=(1e-5, 1e5), arg3:MetricsType="linear", arg4:Optional[dict]=None):
    if not isinstance(arg1, float):
        raise TypeError("arg1 should be a float value")
    
    boundsCheck(arg2)

    arg3Cands=["linear", "additive_chi2", "chi2", "poly", "polynomial", "rbf", "laplacian", "sigmoid", "cosine"]
        
    if not (isinstance(arg3, str) or isinstance(arg3, Callable)):
        raise TypeError(f"arg3 should be a Callable or a string, got {type(arg3).__name__}")
    elif isinstance(arg3, str):
        if arg3 not in arg3Cands and arg3!="precomputed":
            raise ValueError(f"arg3 string value should be one of the following options: {arg3Cands}")
    ## For the case where arg3 is a Callable object
    else:
        pairwiseArg3(arg3)

    ## additive_chi2_kernel do not require any additional parameters.
    if arg4 is not None:
        if not isinstance(arg4, dict):
            raise TypeError(f"arg4 should be a dictionary or None, got {type(arg4).__name__}")
        else:
            if arg3=="linear" or arg3=="cosine":
                targetDic={'dense_output':bool}
                
            elif arg3=="chi2" or arg3=="laplacian" or arg3=="rbf":
                targetDic={'gamma':float}

            elif arg3=="poly" or arg3=="polynomial":
                targetDic={'degree':float, 'gamma':float, 'coef0':float}
            
            elif arg3=="sigmoid":
                targetDic={'gamma':float, 'coef0':float}
            
            elif arg3=="additive_chi2":
                targetDic={}

            else:
                raise ValueError(f"arg3 string value should be one of the following options: {arg3Cands}")
            
            pairwiseArg4(arg4, targetDic)

            if arg3=="laplacian":
                if "gamma" in list(arg4.keys()):
                    gamma=arg4['gamma']
                    if gamma <=0:
                        raise ValueError(f"arg4['gamma'] should be strictly positive.")
    return {'gamma':arg1, 'gamma_bounds':arg2, 'metric':arg3, 'pairwise_kernel_kwargs':arg4}


def rbfArgs(arg1:ScalesType=1.0, arg2:BoundsType=(1e-05, 100000.0)):
    if not (isinstance(arg1, np.ndarray) or isinstance(arg1, float)):
        raise TypeError("arg1 should be a 1D numpy array or a float value.")
    if isinstance(arg1, np.ndarray):
        if arg1.ndim!=1:
            raise ValueError("A numpy array arg1 should be a 1-D array.")
        if arg1.dtype!=np.float64:
            raise TypeError("Elements of numpy array arg1 should be of type np.float64, which is a default float type for numpy arrays.")

    boundsCheck(arg2)
    
    return {'length_scale':arg1, 'length_scale_bounds':arg2}                


def rationalquadraticArgs(arg1:float=1.0, arg2:float=1.0, arg3:BoundsType=(1e-5,1e5), arg4:BoundsType=(1e-5,1e5)):
    if not isinstance(arg1, float):
        raise TypeError("arg1 should be a float value.")
    else:
        if arg1<=0:
            raise ValueError("arg1 should be strictly positive.")
    
    if not isinstance(arg2, float):
        raise TypeError("arg2 hould be a float value.")
    else:
        if arg2<=0:
            raise ValueError("arg2 should be strictly positive.")
    
    boundsCheck(arg3)
    boundsCheck(arg4)

    return {'length_scale':arg1, 'alpha':arg2, 'length_scale_bounds':arg3, 'alpha_bounds':arg4}


def whitekernelArgs(arg1:float=1.0, arg2:BoundsType=(1e-5,1e5)):
    if not isinstance(arg1, float):
        raise TypeError("arg1 should be a float value.")
    else:
        if arg1<0:
            raise ValueError("arg1 should be non-negative.")

    boundsCheck(arg2)

    return {'noise_level':arg1, 'noise_level_bounds':arg2}


    
def argsConstructor(N:int, kernelTypeIdx:list[int], paramsList:list[list]):

    ktiMasks=[kti not in set(kernelTypeDic.keys()) for kti in kernelTypeIdx]
    if any(ktiMasks):
        raise ValueError(f"The kernel type index should be in range(0,9).")
    
    typeKernel=[kernelTypeDic[kti] for kti in kernelTypeIdx]
    paramKernel=[]

    for i, kti in enumerate(kernelTypeIdx):
        paramDic=None
        params=paramsList[i]
        if kti==0:
            paramDic=constantKernelArgs(*params)
        elif kti==1:
            paramDic=dotproductKernelArgs(*params)
        elif kti==2:
            paramDic=expsinesquaredKernelArgs(*params)
        elif kti==3:
            paramDic=exponentiationKernelArgs(*params)
        elif kti==4:
            paramDic=maternKernelArgs(*params)
        elif kti==5:
            paramDic=pairwisekernelArgs(*params)
        elif kti==6:
            paramDic=rbfArgs(*params)
        elif kti==7:
            paramDic=rationalquadraticArgs(*params)
        elif kti==8:
            paramDic=whitekernelArgs(*params)
        else:
            raise ValueError(f"The kernel type index should be in range(0,9), got {kti}.")
        paramKernel.append(paramDic)
    
    return typeKernel, paramKernel


def plotStdInterval(figsize:Tuple[int], X_fit:npt.NDArray, X_pred:npt.NDArray, y_fit:npt.NDArray, y_pred:npt.NDArray,
         y_std:npt.NDArray, xlabel:str, ylabel:str, sigma_coef:float):
    if any([not isinstance(fs, int) for fs in figsize]):
        raise ValueError(f"Elements of figsize should be an int value.")
    if len(figsize)!=2:
        raise ValueError(f"figsize should be of length 2.")
    if not isinstance(X_fit, np.ndarray):
        raise TypeError(f"X_fit should be a 2-D numpy array.")
    if X_fit.ndim!=2:
        raise ValueError(f"X_fit should be a 2-D numpy array.")
    if not isinstance(X_pred, np.ndarray):
        raise TypeError(f"X_pred should be a 2-D numpy array.")
    if X_pred.ndim!=2:
        raise ValueError(f"X_pred should be a 2-D numpy array.")
    if not isinstance(y_fit, np.ndarray):
        raise TypeError(f"y_fit should be a 1-D numpy array.")
    if y_fit.ndim!=1:
        raise ValueError(f"y_fit should be a 1-D numpy array.")
    if not isinstance(y_pred, np.ndarray):
        raise TypeError(f"y_pred should be a 1-D numpy array.")
    if y_pred.ndim!=1:
        raise ValueError(f"y_pred should be a 1-D numpy array.")
    if not isinstance(y_std, np.ndarray):
        raise TypeError(f"y_std should be a 1-D numpy array.")
    if y_std.ndim!=1:
        raise ValueError(f"y_std should be a 1-D numpy array.")
    if X_fit.shape[0]!=y_fit.shape[0]:
        raise ValueError(f"X_fit and y_fit should have equal length in 0-th dimension, got {X_fit.shape[0]} and {y_fit.shape[0]}.")
    if X_pred.shape[0]!=y_pred.shape[0]:
        raise ValueError(f"X_pred and y_pred should have equal length in 0-th dimension, got {X_pred.shape[0]} and {y_pred.shape[0]}.")
    if X_pred.shape[0]!=y_std.shape[0]:
        raise ValueError(f"X_pred and y_std should have equal length in 0-th dimension, got {X_pred.shape[0]} and {y_std.shape[0]}.")
    if sigma_coef<0:
        raise ValueError(f"sigma_coef should be non-negative, got {sigma_coef}.")
    

    plt.figure(figsize=figsize)
    plt.scatter(X_fit, y_fit, c='black', label='Data')
    plt.plot(X_pred, y_pred, label="Prediction")
    plt.fill_between(X_pred.ravel(), y_pred-sigma_coef*y_std, y_pred+sigma_coef*y_std, alpha=0.3, label="Uncertainty")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plotEstim3D(figsize:Tuple[int], gns:npt.NDArray, ubs:npt.NDArray, est:npt.NDArray, 
                xlabel:str="Given Number", ylabel:str="Upper Bound", zlabel:str="Estimate", title:str="Estimates Plot"):
    if any([not isinstance(fs, int) for fs in figsize]):
        raise ValueError(f"Elements of figsize should be an int value.")
    if len(figsize)!=2:
        raise ValueError(f"figsize should be of length 2.")
    if not isinstance(gns, np.ndarray):
        raise TypeError(f"gns should be a 1-D numpy array.")
    if gns.ndim!=1:
        raise ValueError(f"gns should be a 1-D numpy array.")
    if not isinstance(ubs, np.ndarray):
        raise TypeError(f"ubs should be a 1-D numpy array.")
    if ubs.ndim!=1:
        raise ValueError(f"ubs should be a 1-D numpy array.")
    if not isinstance(est, np.ndarray):
        raise TypeError(f"est should be a 1-D numpy array.")
    if est.ndim!=1:
        raise ValueError(f"est should be a 1-D numpy array.")
    if not isinstance(xlabel, str):
        raise TypeError(f"xlabel should be a string value.")
    if not isinstance(ylabel, str):
        raise TypeError(f"ylabel should be a string value.")
    if not isinstance(zlabel, str):
        raise TypeError(f"zlabel should be a string value.")
    if not isinstance(title, str):
        raise TypeError(f"title should be a string value.")
    for idx, gn in enumerate(gns):
        if gn>ubs[idx]:
            raise ValueError(f"given number should be smaller than upper-bound value: got {gn} and {ubs[idx]} at index {idx}")
        if gn<0:
            raise ValueError(f"given number should be non-negative: got {gn} at index {idx}")
        if est[idx]>ubs[idx]:
            raise ValueError(f"estimated number should be smaller than upper-bound value: got {est[idx]} and {ubs[idx]} at index {idx}")
        if est[idx]<0:
            raise ValueError(f"estimated number should be non-negative: got {est[idx]} at index {idx}")
        
    idx=np.arange(len(gns))
    gns_x=interp1d(idx, gns, kind='cubic')
    ubs_y=interp1d(idx, ubs, kind='cubic')
    est_z=interp1d(idx, est, kind='cubic')

    idx_interp=np.linspace(0, len(gns), 5*len(gns))
    gns_interp=gns_x(idx_interp)
    ubs_interp=ubs_y(idx_interp)
    est_interp=est_z(idx_interp)

    fig=plt.figure(figsize=figsize)
    ax=fig.add_subplot(111, projection='3d')
    ax.plot(gns_interp, ubs_interp, est_interp, color='black')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    plt.show()


def plotFreq3D(figsize:Tuple[int], gns:npt.NDArray, ubs:npt.NDArray, 
               xlabel:str="Given Number", ylabel:str="Upper Bound", zlabel:str="Frequency", 
               title:str="Design selection frequencies"):
    
    if any([not isinstance(fs, int) for fs in figsize]):
        raise ValueError(f"Elements of figsize should be an int value.")
    if len(figsize)!=2:
        raise ValueError(f"figsize should be of length 2.")
    if not isinstance(gns, np.ndarray):
        raise TypeError(f"gns should be a 1-D numpy array.")
    if gns.ndim!=1:
        raise ValueError(f"gns should be a 1-D numpy array.")
    if not isinstance(ubs, np.ndarray):
        raise TypeError(f"ubs should be a 1-D numpy array.")
    if ubs.ndim!=1:
        raise ValueError(f"ubs should be a 1-D numpy array.")
    if not isinstance(xlabel, str):
        raise TypeError(f"xlabel should be a string value.")
    if not isinstance(ylabel, str):
        raise TypeError(f"ylabel should be a string value.")
    if not isinstance(zlabel, str):
        raise TypeError(f"zlabel should be a string value.")
    if not isinstance(title, str):
        raise TypeError(f"title should be a string value.")
    for idx, gn in enumerate(gns):
        if gn>ubs[idx]:
            raise ValueError(f"given number should be smaller than upper-bound value: got {gn} and {ubs[idx]} at index {idx}")
        if gn<0:
            raise ValueError(f"given number should be non-negative: got {gn} at index {idx}")
       
    fig=plt.figure(figsize=figsize)
    ax=fig.add_subplot(projection='3d')
    hist, gn_edge, ub_edge=np.histogram2d(gns, ubs, bins=[10, 50], range=[[0, 500], [0, 500]])

    gn_pos, ub_pos=np.meshgrid(gn_edge[:-1], ub_edge[:-1], indexing="ij")
    gn_pos=gn_pos.ravel()
    ub_pos=ub_pos.ravel()
    freq_pos=0

    w=50*np.ones_like(freq_pos)
    d=10*np.ones_like(freq_pos)
    h=hist.ravel()
    
    ax.bar3d(gn_pos, ub_pos, freq_pos, w, d, h, zsort='average')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)

    plt.show()
    