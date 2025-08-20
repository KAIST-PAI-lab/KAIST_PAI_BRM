from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *
import numpy as np
from functools import reduce
from typing import Optional
from typing import Union
import numpy.typing as npt
import re


def GPRInstance(kernel_type_strs:list[str], kernel_arguments_dics:list[dict], combine_format:str, 
                alpha:Union[float, npt.NDArray[np.float64]]=1e-10, 
                normalize_y:bool=True, n_restarts_optimizer:int=0, 
                random_state:Optional[Union[int, np.random.RandomState]]=None):
    if not isinstance(combine_format, str):
        raise TypeError(f"combine_format should be a string, got {type(format).__name__}.")
    if not isinstance(alpha, Union[float, np.ndarray]):
        raise TypeError(f"alpha should be float or numpy array, got {type(alpha).__name__}.")
    if isinstance(alpha, float):
        if alpha<=0:
            raise ValueError(f"alpha should be a positive value, got {alpha}.")
    if isinstance(alpha, np.ndarray):
        if alpha.dtype!=np.float64:
            raise TypeError(f"alpha should have a dtype of np.float64.")
        if not np.all(alpha>0):
            raise ValueError(f"alpha should only contain positive elements.")
    if not isinstance(normalize_y, bool):
        raise TypeError(f"normalize_y should be a bool value, got {type(normalize_y).__name__}.")
    if not isinstance(n_restarts_optimizer, int):
        raise TypeError(f"n_restarts_optimizer should be an integer value, got {type(n_restarts_optimizer).__name__}.")
    if n_restarts_optimizer<0:
        raise ValueError(f"n_restarts_optimizer should be non-negative, got {n_restarts_optimizer}.")
    if random_state is not None:
        if not (isinstance(random_state, int) or isinstance(random_state, np.random.RandomState)):
            raise TypeError(f"random_state should be None, an int value, or a RandomState object. Got {type(random_state).__name__}")


    kBuilder=KernelBuilder(num_kernels=len(kernel_type_strs), 
                           kernel_types_list=kernel_type_strs, 
                           kernel_arguments_dics_list=kernel_arguments_dics, 
                           combine_format_str=combine_format)
    Kernel=kBuilder.create_compound_kernel()
    gpr=GaussianProcessRegressor(kernel=Kernel, alpha=alpha, normalize_y=normalize_y, 
                                 n_restarts_optimizer=n_restarts_optimizer, random_state=random_state)
    return Kernel, gpr


## Adding and multiplying kernel instances are processed after creating all individual kernel instances.
class KernelBuilder():
    def __init__(self, num_kernels:int, kernel_types_list:list[str], kernel_arguments_dics_list:list[dict], combine_format_str:str):
        if not isinstance(num_kernels, int):
            raise TypeError(f"num_kernels should be an int value, got the type of {type(num_kernels).__name__}.")
        if num_kernels<=0:
            raise ValueError(f"num_kernels should be a positive integer, got {num_kernels}.")
        if not isinstance(kernel_types_list, list):
            raise TypeError(f"kernel_types_list should be a list, got the type of {type(kernel_types_list).__name__}.")
        if not all(isinstance(kernel, str) for kernel in kernel_types_list):
            typ=kernel_types_list[[isinstance(kernel, str) for kernel in kernel_types_list].index(False)]
            raise TypeError(f"kernel_types_list should contain string elements, got a {type(typ).__name__ } type element.")
        if len(kernel_types_list)!=num_kernels:
            raise ValueError(f"kernel_types_list should be of length {num_kernels}, {len(kernel_types_list)}.")
        if not isinstance(kernel_arguments_dics_list, list):
            raise TypeError(f"kernel_arguments_dics_list should be a list, got the type of {type(kernel_arguments_dics_list).__name__}.")
        if not all(isinstance(kad, dict) for kad in kernel_arguments_dics_list):
            typ=kernel_arguments_dics_list[[isinstance(kad, dict) for kad in kernel_arguments_dics_list].index(False)]
            raise TypeError(f"kernel_arguments_dics_list should contain dictionary elements, got a {type(typ).__name__} type element.")
        if len(kernel_arguments_dics_list)!=num_kernels:
            raise ValueError(f"kernel_arguments_dics_list should be of length {num_kernels}, got {len(kernel_arguments_dics_list)}.")
        if not isinstance(combine_format_str, str):
            raise TypeError(f"combine_format_str should be a string, got the type of {type(combine_format_str).__name__}.")
        
        split_format=re.split(r'(\+|\*)', combine_format_str)
        if len(split_format) != 2*num_kernels-1:
            raise ValueError(f"combine_format should indicate a valid combination of kernel objects.")
        for i in range(0, (len(split_format)+1)//2):
            if int(split_format[2*i][1:])!=i+1:
                raise ValueError(f"The {i}-th operand of the format should be k{i+1}, not {split_format[2*i]}.")
            
        numbers_format=re.findall(r'\d+', combine_format_str)
        kernel_symbol_idxs=list(map(int, numbers_format))
        if max(kernel_symbol_idxs)!=num_kernels:
            raise ValueError(f"combine_format should include {num_kernels} kernel symbols (from k1 to k{num_kernels}), got {max(kernel_symbol_idxs)} symbols.")
        
        chars_format=re.search(r'[^a-zA-z0-9+*]', combine_format_str)
        if chars_format:
            raise ValueError(f"combine_format should only include alphabet characters, numbers, +, and *.")
        

        self.num_kernels=num_kernels
        self.kernel_types_list=kernel_types_list
        self.kernel_arguments_dics_list=kernel_arguments_dics_list
        self.split_format=split_format

        self.basicTypes=['ConstantKernel', 'DotProduct', 'ExpSineSquared', 'Exponentiation',
                        'Matern', 'PairwiseKernel', 'RBF', 'RationalQuadratic', 'WhiteKernel']
        self.kernel: Optional[Kernel] = None
        self.individual_kernels=[]
    
    def create_compound_kernel(self):
        if self.num_kernels==1:
            kernel=self.create_individual_kernel(kernel_type_str=self.kernel_types_list[0], 
                                                 kernel_args_dict=self.kernel_arguments_dics_list[0])    
            self.kernel=kernel
        else:
            for kIdx in range(self.num_kernels):
                kernel_element=self.create_individual_kernel(kernel_type_str=self.kernel_types_list[kIdx], 
                                                             kernel_args_dict=self.kernel_arguments_dics_list[kIdx])
                self.individual_kernels.append(kernel_element)

            for eIdx, format_elem in enumerate(self.split_format):
                if format_elem=="*":  # * is always at the odd index
                    operand_first=self.individual_kernels[(eIdx-1)//2]
                    operand_second=self.individual_kernels[(eIdx+1)//2]
                    kernel_product=operand_first*operand_second
                    self.individual_kernels.insert((eIdx-1)//2, kernel_product)
                    self.individual_kernels.remove(operand_first)
                    self.individual_kernels.remove(operand_second)
                else:
                    continue
            compound_kernel=reduce(lambda x,y:x+y, self.individual_kernels)
            self.kernel=compound_kernel
        return self.kernel
    
    def create_individual_kernel(self, kernel_type_str:str, kernel_args_dict:dict):
        def check_params_match(kernel:Kernel, given_args_dict:dict):
            kernel_params=kernel.get_params().keys()
            if not set(given_args_dict.keys()).issubset(set(kernel_params)):
                raise ValueError(f"The parameters provided ({set(given_args_dict.keys())}) do not match the required parameters: {set(kernel_params)}.")
            return True

        if not isinstance(kernel_type_str, str):
            raise TypeError(f"kernel_type_str should be a string value, got the type of {type(kernel_type_str).__name__}.")
        if not isinstance(kernel_args_dict, dict):
            raise TypeError(f"kernel_args_dict should be a dictionary, got the type of {type(kernel_args_dict).__name__}.")
        
        kernel_index=0
        individual_kernel=None
        try:
            kernel_index=self.basicTypes.index(kernel_type_str)
        except ValueError:
            raise ValueError(f"The given kernel type '{kernel_type_str}' is not a valid kernel type.")
        
        if kernel_index==0:
            individual_kernel=ConstantKernel()
            if check_params_match(kernel=individual_kernel, given_args_dict=kernel_args_dict):
                individual_kernel.set_params(**kernel_args_dict)

        elif kernel_index==1:
            individual_kernel=DotProduct()
            if check_params_match(kernel=individual_kernel, given_args_dict=kernel_args_dict):
                individual_kernel.set_params(**kernel_args_dict)
        
        elif kernel_index==2:
            individual_kernel=ExpSineSquared()
            if check_params_match(kernel=individual_kernel, given_args_dict=kernel_args_dict):
                individual_kernel.set_params(**kernel_args_dict)

        elif kernel_index==3:
            if len(kernel_args_dict.keys())!=2:
                raise ValueError(f"The Exponentiation kernel takes 2 arguments, got {len(kernel_args_dict.keys())} arguments.")
            if not isinstance(kernel_args_dict['kernel'], Kernel):
                raise TypeError(f'''The 'kernel' argument of the Exponentiation kernel should be a valid kernel instance, 
                                got the type of {type(kernel_args_dict['kernel']).__name__}.''')
            if not isinstance(kernel_args_dict['exponent'], float):
                raise TypeError(f'''The 'exponent' argument of the Exponentiation kernel should be a float value, 
                                got the type of {type(kernel_args_dict['exponent']).__name__}.''')
            individual_kernel=Exponentiation(**kernel_args_dict)

        elif kernel_index==4:
            individual_kernel=Matern()
            if check_params_match(kernel=individual_kernel, given_args_dict=kernel_args_dict):
                individual_kernel.set_params(**kernel_args_dict)
        
        elif kernel_index==5:
            individual_kernel=PairwiseKernel()
            if check_params_match(kernel=individual_kernel, given_args_dict=kernel_args_dict):
                individual_kernel.set_params(**kernel_args_dict)

        elif kernel_index==6:
            individual_kernel=RBF()
            if check_params_match(kernel=individual_kernel, given_args_dict=kernel_args_dict):
                individual_kernel.set_params(**kernel_args_dict)

        elif kernel_index==7:
            individual_kernel=RationalQuadratic()
            if check_params_match(kernel=individual_kernel, given_args_dict=kernel_args_dict):
                individual_kernel.set_params(**kernel_args_dict)

        elif kernel_index==8:
            individual_kernel=WhiteKernel()
            if check_params_match(kernel=individual_kernel, given_args_dict=kernel_args_dict):
                individual_kernel.set_params(**kernel_args_dict)
        
        else:
            raise ValueError(f"The type '{kernel_type_str}' is not a valid kernel type.")

        return individual_kernel
