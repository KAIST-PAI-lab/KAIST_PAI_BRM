from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *
import numpy as np
from functools import reduce
from typing import Optional

## nKernel: The Number of kernel instances to be combined
## typeKernel: List of types of each kernel instance
## paramKernel: List of parameters for each kernel instance
def GPRInstance(nKernel:int, typeKernel:list[str], paramKernel:list[dict], mulIdx:list[list[int]], sumIdx:list[list[int]], alpha=1e-10, normalize_y=True, n_restarts_optimizer=0, random_state=None):
    kernel=KernelInstance(nKernel, typeKernel, paramKernel, mulIdx, sumIdx)
    gpr=GaussianProcessRegressor(kernel, alpha=alpha, normalize_y=normalize_y, n_restarts_optimizer=n_restarts_optimizer, random_state=random_state)
    return kernel, gpr


## Adding and multiplying kernel instances are processed after creating all individual kernel instances.
class KernelInstance(Kernel):
    def __init__(self, nKernel:int, typeKernel:list[str], paramKernel:list[dict], mulIdx:list[list[int]], sumIdx:list[list[int]]):
        assert len(typeKernel)==nKernel, "The number of kernel instances to be combined do not match the length of a list of their types."
        assert len(paramKernel)==nKernel, "The number of kernel instances to be combined do not match the length of a list of their parameters."
        assert nKernel>0, "The number of kernel instances to be combined should be larger than 0"
        assert len(mulIdx)==2, "The mulIdx list should have 2 list elements"
        assert len(mulIdx[0])==len(mulIdx[1]), "The lists in mulIdx list should be of the same length"
        assert len(sumIdx)==2, "The sumIdx list should have 2 list elements"
        assert len(sumIdx[0])==len(sumIdx[1]), "The lists in sumIdx list should be of the same length"

        self.nKernel=nKernel
        self.typeKernel=typeKernel
        self.paramKernel=paramKernel
        self.mulIdx=mulIdx
        self.sumIdx=sumIdx

        self.basicTypes=['ConstantKernel', 'DotProduct', 'ExpSineSquared', 'Exponentiation',
                        'Matern', 'PairwiseKernel', 'RBF', 'RationalQuadratic', 'WhiteKernel']
        self.kernel: Optional[Kernel] = None
        self.kernels=[]

        if nKernel==1:
            self.kernel=self.create(typeKernel[0], paramKernel[0])    
        
        else:
            for n in range(nKernel):
                kernelElem=self.create(typeKernel[n], paramKernel[n])
                self.kernels.append(kernelElem)

            for mn in range(len(mulIdx[0])):
                ## Since two kernels will be popped and one kernel (mulKernel) will be inserted
                ## The index of the operand kernel decrements for each iteration
                idx1=mulIdx[0][mn]-mn
                idx2=mulIdx[1][mn]-mn

                k1=self.kernels[idx1]  
                k2=self.kernels[idx2]
                mulKernel=Product(k1, k2)
                
                self.kernels.pop(idx2)
                self.kernels.pop(idx1)
                self.kernels.insert(idx1, mulKernel)

            kernel=reduce(lambda k1, k2: k1+k2, self.kernels)
        self.kernel=kernel
    
    def create(self, type:str, params:dict):
        def matchParam(kernel:Kernel, params:dict):
            kernelKeys=kernel.get_params().keys()
            if set(params.keys())!=set(kernelKeys):
                raise TypeError(f"The parameters provided ({set(params.keys())}) do not match the required ones: {set(kernelKeys)}.")
            return True

        kernelIdx=0
        kernelInstance=None
        try:
            kernelIdx=self.basicTypes.index(type)
        except ValueError:
            raise ValueError(f"The type '{type}' is not a valid kernel type.")
        
        '''
        if kernelIdx==0:
            if not all(isinstance(p, Kernel) for p in params[0]):
                raise TypeError("The parameters for CompoundKernel should be a valid Kernel instances.")
            kernelInstance=CompoundKernel(**params)
        '''
        if kernelIdx==0:
            kernelInstance=ConstantKernel(constant_value_bounds="fixed")
            if matchParam(kernelInstance, params):
                kernelInstance.set_params(**params)

        elif kernelIdx==1:
            kernelInstance=DotProduct(sigma_0_bounds="fixed")
            if matchParam(kernelInstance, params):
                kernelInstance.set_params(**params)
        
        elif kernelIdx==2:
            kernelInstance=ExpSineSquared(length_scale_bounds="fixed", periodicity_bounds="fixed")
            if matchParam(kernelInstance, params):
                kernelInstance.set_params(**params)

        elif kernelIdx==3:
            assert len(params.keys())==2, "The Exponentiation kernel takes 2 arguments."
            assert isinstance(params[0], Kernel), "The first argument of the Exponentiation kernel should be a valid kernel instance."
            assert isinstance(params[1], int), "The second argument of the Exponentiation kernel should be an integer."
            kernelInstance=Exponentiation(**params)

        elif kernelIdx==4:
            kernelInstance=Matern(length_scale_bounds="fixed")
            if matchParam(kernelInstance, params):
                kernelInstance.set_params(**params)
        
        elif kernelIdx==5:
            kernelInstance=PairwiseKernel(gamma_bounds="fixed")
            if matchParam(kernelInstance, params):
                kernelInstance.set_params(**params)

        
        elif kernelIdx==6:
            kernelInstance=RBF(length_scale_bounds="fixed")
            if matchParam(kernelInstance, params):
                kernelInstance.set_params(**params)

        elif kernelIdx==7:
            kernelInstance=RationalQuadratic(length_scale_bounds="fixed", alpha_bounds="fixed")
            if matchParam(kernelInstance, params):
                kernelInstance.set_params(**params)

        elif kernelIdx==8:
            kernelInstance=WhiteKernel(noise_level_bounds="fixed")
            if matchParam(kernelInstance, params):
                kernelInstance.set_params(**params)
        
        else:
            raise ValueError(f"The type '{type}' is not a valid basic kernel type.")

        return kernelInstance
    
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
    