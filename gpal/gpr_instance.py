from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *
import numpy as np
from functools import reduce
from typing import Optional
from typing import Union
from typing import List
import numpy.typing as npt

## nKernel: The Number of kernel instances to be combined
## typeKernel: List of types of each kernel instance
## paramKernel: List of parameters for each kernel instance
def GPRInstance(nKernel:int, typeKernel:list[str], paramKernel:list[dict], mulIdx:list[list[int]], sumIdx:list[list[int]], 
                alpha:Union[float, npt.NDArray[np.float64]]=1e-10, normalize_y:bool=True, n_restarts_optimizer:int=0, 
                random_state:Optional[Union[int, np.random.RandomState]]=None):
    if not isinstance(alpha, Union[float, np.ndarray]):
        raise TypeError(f"alpha should be float or numpy array: got {type(alpha).__name__}.")
    if isinstance(alpha, float):
        if alpha<=0:
            raise ValueError(f"alpha should be positive.")
    if isinstance(alpha, np.ndarray):
        if alpha.dtype!=np.float64:
            raise TypeError(f"alpha should have a dtype of np.float64.")
        if not np.all(alpha>0):
            raise ValueError(f"alpha should contain positive elements.")
    if not isinstance(normalize_y, bool):
        raise TypeError(f"normalize_y should be a bool value.")
    if not isinstance(n_restarts_optimizer, int):
        raise TypeError(f"n_restarts_optimizer should be an int value.")
    if n_restarts_optimizer<0:
        raise ValueError(f"n_restarts_optimizer should be non-negative.")
    if random_state is not None:
        if not (isinstance(random_state, int) or isinstance(random_state, np.random.RandomState)):
            raise TypeError(f"random_state should be a None, an int value, or a RandomState object.")


    kBuilder=KernelBuilder(nKernel, typeKernel, paramKernel, mulIdx, sumIdx)
    kernel=kBuilder.create_compound()
    gpr=GaussianProcessRegressor(kernel, alpha=alpha, normalize_y=normalize_y, n_restarts_optimizer=n_restarts_optimizer, random_state=random_state)
    return kernel, gpr


## Adding and multiplying kernel instances are processed after creating all individual kernel instances.
class KernelBuilder():
    def __init__(self, nKernel:int, typeKernel:list[str], paramKernel:list[dict], mulIdx:list[list[int]], sumIdx:list[list[int]]):
        if not isinstance(nKernel, int):
            raise TypeError(f"nKernel should be an int value, got {type(nKernel).__name__}.")
        if nKernel<=0:
            raise ValueError(f"nKernel should be positive.")
        if not isinstance(typeKernel, list):
            raise TypeError(f"typeKernel should be a list, got {type(typeKernel).__name__}.")
        if not all(isinstance(tk, str) for tk in typeKernel):
            typ=typeKernel[[isinstance(tk, str) for tk in typeKernel].index(False)]
            raise TypeError(f"typeKernel should contain string values, got {type(typ).__name__ }.")
        if len(typeKernel)!=nKernel:
            raise ValueError(f"typeKernel should be of length {nKernel}, got {len(typeKernel)}.")
        if not isinstance(paramKernel, list):
            raise TypeError(f"paramKernel should be a list, got {type(paramKernel).__name__}.")
        if not all(isinstance(pk, dict) for pk in paramKernel):
            typ=paramKernel[[isinstance(pk, dict) for pk in paramKernel].index(False)]
            raise TypeError(f"paramKernel should contain dict values, got {type(typ).__name__}.")
        if len(paramKernel)!=nKernel:
            raise ValueError(f"paramKernel should be of length {nKernel}, got {len(paramKernel)}.")
        if not isinstance(mulIdx, list):
            raise TypeError(f"mulIdx should be a list, got {type(mulIdx).__name__}.")
        if len(mulIdx)!=2:
            raise ValueError(f"mulIdx should contain 2 lists, got {len(mulIdx)}.")
        if not isinstance(mulIdx[0], list):
            raise TypeError(f"mulIdx should contain lists, got {type(mulIdx[0]).__name__} at 0-th index.")
        if not isinstance(mulIdx[1], list):
            raise TypeError(f"mulIdx should contain lists, got {type(mulIdx[1]).__name__} at 1-th index.")
        if len(mulIdx[0])!=len(mulIdx[1]):
            raise ValueError(f"mulIdx should contain two lists of same length, got {len(mulIdx[0])} and {len(mulIdx[1])}.")
        if not all(isinstance(mi, int) for mi in mulIdx[0]):
            mIdx0=mulIdx[0]
            idx=[isinstance(mi, int) for mi in mIdx0].index(False)
            typ=mIdx0[idx]
            raise TypeError(f"mulIdx should contain 2 lists with integer elements, got {type(typ).__name__} at {idx}-th index of mulIdx[0].")
        if not all(isinstance(mi, int) for mi in mulIdx[1]):
            mIdx1=mulIdx[1]
            idx=[isinstance(mi, int) for mi in mIdx1].index(False)
            typ=mIdx1[idx]
            raise TypeError(f"mulIdx should contain 2 lists with integer elements, got {type(typ).__name__} at {idx}-th index of mulIdx[1].")
        if not isinstance(sumIdx, list):
            raise TypeError(f"sumIdx should be a list, got {type(sumIdx).__name__}.")
        if len(sumIdx)!=2:
            raise ValueError(f"sumIdx should contain 2 lists, got {len(sumIdx)}.")
        if not isinstance(sumIdx[0], list):
            raise TypeError(f"sumIdx should contain lists, got {type(sumIdx[0]).__name__} at 0-th index.")
        if not isinstance(sumIdx[1], list):
            raise TypeError(f"sumIdx should contain lists, got {type(sumIdx[1]).__name__} at 1-th index.")
        if len(sumIdx[0])!=len(sumIdx[1]):
            raise ValueError(f"sumIdx should contain two lists of same length, got {len(sumIdx[0])} and {len(sumIdx[1])}.")
        if not all(isinstance(si, int) for si in sumIdx[0]):
            sIdx0=sumIdx[0]
            idx=[isinstance(si, int) for si in sIdx0].index(False)
            typ=sIdx0[idx]
            raise TypeError(f"sumIdx should contain 2 lists with integer elements, got {type(typ).__name__} at {idx}-th index of sumIdx[0].")
        if not all(isinstance(si, int) for si in sumIdx[1]):
            sIdx1=sumIdx[1]
            idx=[isinstance(si, int) for si in sIdx1].index(False)
            typ=sIdx1[idx]
            raise TypeError(f"sumIdx should contain 2 lists with integer elements, got {type(typ).__name__} at {idx}-th index of sumIdx[1].")
        

        self.nKernel=nKernel
        self.typeKernel=typeKernel
        self.paramKernel=paramKernel
        self.mulIdx=mulIdx
        self.sumIdx=sumIdx

        self.basicTypes=['ConstantKernel', 'DotProduct', 'ExpSineSquared', 'Exponentiation',
                        'Matern', 'PairwiseKernel', 'RBF', 'RationalQuadratic', 'WhiteKernel']
        self.kernel: Optional[Kernel] = None
        self.kernels=[]
    
    def create_compound(self):
        if self.nKernel==1:
            kernel=self.create(self.typeKernel[0], self.paramKernel[0])    
        else:
            for n in range(self.nKernel):
                kernelElem=self.create(self.typeKernel[n], self.paramKernel[n])
                self.kernels.append(kernelElem)

            for mn in range(len(self.mulIdx[0])):
                ## Since two kernels will be popped and one kernel (mulKernel) will be inserted
                ## The index of the operand kernel decrements for each iteration
                idx1=self.mulIdx[0][mn]-mn
                idx2=self.mulIdx[1][mn]-mn

                k1=self.kernels[idx1]  
                k2=self.kernels[idx2]
                mulKernel=Product(k1, k2)
                
                self.kernels.pop(idx2)
                self.kernels.pop(idx1)
                self.kernels.insert(idx1, mulKernel)

            kernel=reduce(lambda k1, k2: k1+k2, self.kernels)

        self.kernel=kernel
        return self.kernel
    
    def create(self, typeK:str, params:dict):
        def matchParam(kernel:Kernel, params:dict):
            kernelKeys=kernel.get_params().keys()
            if set(params.keys())!=set(kernelKeys):
                raise ValueError(f"The parameters provided ({set(params.keys())}) do not match the required ones: {set(kernelKeys)}.")
            return True

        if not isinstance(typeK, str):
            raise TypeError(f"typeK should be a string value, got {type(typeK).__name__}.")
        if not isinstance(params, dict):
            raise TypeError(f"params should be a dictionary, got {type(params).__name__}.")
        kernelIdx=0
        kernelInstance=None
        try:
            kernelIdx=self.basicTypes.index(typeK)
        except ValueError:
            raise ValueError(f"The type '{typeK}' is not a valid kernel type.")
        
        if kernelIdx==0:
            kernelInstance=ConstantKernel()
            if matchParam(kernelInstance, params):
                kernelInstance.set_params(**params)

        elif kernelIdx==1:
            kernelInstance=DotProduct()
            if matchParam(kernelInstance, params):
                kernelInstance.set_params(**params)
        
        elif kernelIdx==2:
            kernelInstance=ExpSineSquared()
            if matchParam(kernelInstance, params):
                kernelInstance.set_params(**params)

        elif kernelIdx==3:
            if len(params.keys())!=2:
                raise ValueError(f"The Exponentiation kernel takes 2 arguments, got {len(params.keys())}.")
            if not isinstance(params[0], Kernel):
                raise TypeError(f"The first argument of the Exponentiation kernel should be a valid kernel instance, got {type(params[0]).__name__}.")
            if not isinstance(params[1], int):
                raise TypeError(f"The second argument of the Exponentiation kernel should be an integer, got {type(params[1]).__name__}.")
            kernelInstance=Exponentiation(**params)

        elif kernelIdx==4:
            kernelInstance=Matern()
            if matchParam(kernelInstance, params):
                kernelInstance.set_params(**params)
        
        elif kernelIdx==5:
            kernelInstance=PairwiseKernel()
            if matchParam(kernelInstance, params):
                kernelInstance.set_params(**params)

        elif kernelIdx==6:
            kernelInstance=RBF()
            if matchParam(kernelInstance, params):
                kernelInstance.set_params(**params)

        elif kernelIdx==7:
            kernelInstance=RationalQuadratic()
            if matchParam(kernelInstance, params):
                kernelInstance.set_params(**params)

        elif kernelIdx==8:
            kernelInstance=WhiteKernel()
            if matchParam(kernelInstance, params):
                kernelInstance.set_params(**params)
        
        else:
            raise ValueError(f"The type '{typeK}' is not a valid basic kernel type.")

        return kernelInstance
    '''
    def __call__(self, X, Y=None, eval_gradient=False):
        return self.kernel(X, Y, eval_gradient)
    
    def diag(self, X):
        return self.kernel.diag(X)
    
    def is_stationary(self):
        return self.kernel.is_stationary()

    
    def get_params(self, deep=True):
        return {"nKernel":self.nKernel, 
                "typeKernel":self.typeKernel, 
                "paramKernel":self.paramKernel,
                "mulIdx":self.mulIdx, 
                "sumIdx":self.sumIdx}
    '''