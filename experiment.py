import os
import argparse
import numpy as np
import warnings
import random
import pandas as pd
import pickle
from datetime import datetime
from sklearn.gaussian_process import GaussianProcessRegressor


from psychopy import visual, core, event, gui
from psychopy.visual import ElementArrayStim
# Suppress Psychopy logging messages
from psychopy import logging
logging.console.setLevel(logging.ERROR)


from nlt.gpal.argparser import argparser
from nlt.gpal.gpr_instance import *
from nlt.gpal.utils import prediction, argsConstructor, plotStdInterval, plotEstim3D, plotFreq3D
from nlt.nlt_main.utils.utils import collect_participant_info, save_results
from nlt.nlt_main.NLE.draw_dots import draw_grid_position, calculate_dot_size
from nlt.nlt_main.main import *

os.environ["CUDA_VISIBLE_DEVICES"] = '2'  # Specifying the GPU to use




if __name__=="__main__":

    args=argparser()

    ## kernelTypeDic={0:'ConstantKernel', 
    ##                1:'DotProduct', 
    ##                2:'ExpSineSquared', 
    ##                3:'Exponentiation',
    ##                4:'Matern', 
    ##                5:'PairwiseKernel', 
    ##                6:'RBF', 
    ##                7:'RationalQuadratic', 
    ##                8:'WhiteKernel'}


    ######################################################################################################
    ########################################### Modify these values ######################################
    ######################################################################################################

    subject_id='0'                                      # subject ID

    args.n_trials=45                            # Number of experiment trials for a subject
    args.seed=211                               # A random seed value for reproducibility
    
    ## Arguments related to initializing the GPR instance.

    args.n_kernels=2                            # The number of individual kernels to be combined. Should be a positive integer.
    args.alpha=1e-10                            # A value added to the diagonal of the kernel matrix during fitting.
    args.normalize_y=True                       # A binary mask indicating whether to normalize the target values while fitting.
    args.n_restarts_optimizer=0                 # The number of restarts of the optimizer to find the optimal kernel parameters.
    args.type_kernels_index=[0,6]               # A list of indices of kernels to be combined. Refer to 'kernelTypeDic' above.
    args.parameters_list=[[1.0, 'fixed'],       # A list of list of arguments to be fed to each kernel.
                          [1.0, 'fixed']]
    args.multiplied_indices=[[0,1]]             # A list of lists indicating the kernels to be multiplied.
    args.summed_indices=[[]]                    # A list of lists indicating the kernels to be summed.
    args.gpr_random_state=None                  # A parameter determining random number generation in initializing the centers of the GP regressor.
    
    ## Arguments related to optimizing the GPR instance.

    args.return_std=True                        # A binary mask indicating whether to return standard deviation of posterior distribution at each query value.
    args.return_cov=False                       # A binary mask indicating whether to return covaraince matrix of posterior distribution at each query value.
    args.optim_mode='GPAL'                      # Optimization algorithm to use. Should be "GPAL" or "ADO".
    args.base_model=None                        # A base model for ADO optimization.
    
    ## Arguments related to running an experiment.

    args.save_results_dir='../saved_results'    # A directory to store the task results.
    args.save_models_dir='../saved_models'      # A directory to store the trained Gaussian process regressor models.
    args.enable_gpu=False                       # A binary mask indicating whether to use GPU.
    args.subject_prefix='Subject'               # A prefix attached to the unique subject ID to construct a full indicator of each subject.


    ## Arguments related to plotting

    ## Figure 1: Given number estimates and associated standard deviation 
    args.figure_size_1=(8,6)                    # The size of the figure #1.
    args.label_x_1='Given Number'               # A label for the x-axis of the figure #1.
    args.label_y_1='Number Estimate'            # A label for the y-axis of the figure #1.
    args.title_1='Given Number Estimates (2D)'  # A title of the figure #1.
    args.sigma_coef_1=1.0                       # A coefficient multiplied to the standard deviation, determining the range of uncertainty to be plotted.

    ## Figure 2: 3D plot of given number estimates for 2D number line task.
    args.figure_size_2=(8,6)                    # The size of the figure #2.
    args.label_x_2='Given Number'               # A label for the x-axis of the figure #2.
    args.label_y_2='Upper Bound'                # A label for the y-axis of the figure #2.
    args.label_z_2='Given Number Estimates'     # A label for the z-axis of the figure #2.
    args.title_2='Given Number Estimates (3D)'  # A title of the figure #2.
    
    ## Figure 3: A histogram illustrating the frequencies each design was selected by the optimization algorithm.
    args.figure_size_3=(8,6)                    # The size of the figure #3.
    args.label_x_3='Given Number'               # A label for the x-axis of the figure #3.
    args.label_y_3='Upper Bound'                # A label for the y-axis of the figure #3.
    args.label_z_3='Frequency'     # A label for the z-axis of the figure #3.
    args.title_3='Design Selection Frequencies'  # A title of the figure #3.


    ## Arguments related to ADO optimization.


    ######################################################################################################
    ########################################### Modify these values ######################################
    ######################################################################################################

    # warnings.filterwarnings("ignore")
    random.seed(args.seed)
    
    sbj=args.subject_prefix+str(subject_id)
    save_models_sbj_dir=args.save_models_dir+'/'+sbj  # The directory to save the optimized model for each subject

    type_kernels, param_dics=argsConstructor(args.n_kernel, args.type_kernels_index, args.parameters_list)
    #kernelBuilder=KernelBuilder(args.n_kernel, type_kernels, param_dics, args.multiplied_indices, args.summed_indices)
    #kernel=kernelBuilder.create_compound()
    kernel, gpr=GPRInstance(args.n_kernel, type_kernels, param_dics, args.multiplied_indices, args.summed_indices, 
                            alpha=args.alpha, n_restarts_optimizer=args.n_restarts_optimizer, 
                            normalize_y=args.normalize_y, random_state=args.gpr_random_state)
    
    run_experiment(args, gpr, subject_id)
    