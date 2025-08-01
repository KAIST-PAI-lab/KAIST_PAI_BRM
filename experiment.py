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


from gpal.argparser import argparser
from gpal.gpr_instance import *
from gpal.utils import argsConstructor, plotStd2D, plotEstim3D, plotFreq3D
from nlt_main.utils.utils import collect_participant_info
#from nlt_main.NLE.draw_dots import draw_grid_position, calculate_dot_size
from nlt_main.main import run_experiment

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

    #subject_id='0'                                      # subject ID

    args.n_trials=45                           # Number of experiment trials for a subject
    args.seed=211                               # A random seed value for reproducibility
    args.n_DVs=2
    ## Arguments related to initializing the GPR instance.

    args.n_kernels=2                             # The number of individual kernels to be combined. Should be a positive integer.
    args.alpha=1e-10                            # A value added to the diagonal of the kernel matrix during fitting.
    args.normalize_y=True                       # A binary mask indicating whether to normalize the target values while fitting.
    args.n_restarts_optimizer=0                 # The number of restarts of the optimizer to find the optimal kernel parameters.
    args.type_kernels_index=[0,6]               # A list of indices of kernels to be combined. Refer to 'kernelTypeDic' above.
    args.parameters_list=[[1.0, 'fixed'],       # A list of list of arguments to be fed to each kernel.
                          [1.0, 'fixed']]
    args.multiplied_indices=[[0], [1]]             # A list of lists indicating the kernels to be multiplied.
    args.summed_indices=[[], []]                    # A list of lists indicating the kernels to be summed.
    args.gpr_random_state=None                  # A parameter determining random number generation in initializing the centers of the GP regressor.
    
    ## Arguments related to optimizing the GPR instance.

    args.return_std=True                        # A binary mask indicating whether to return standard deviation of posterior distribution at each query value.
    args.return_cov=False                       # A binary mask indicating whether to return covaraince matrix of posterior distribution at each query value.
    args.optim_mode='GPAL'                      # Optimization algorithm to use. Should be "GPAL" or "ADO".
    args.base_model=None                        # A base model for ADO optimization.
    
    ## Arguments related to running an experiment.

    args.save_results_dir='results'    # A directory to store the task results.
    args.save_models_dir='models'      # A directory to store the trained Gaussian process regressor models.
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
    
    
    type_kernels, param_dics=argsConstructor(args.n_kernels, args.type_kernels_index, args.parameters_list)
    kernel, gpr=GPRInstance(args.n_kernels, type_kernels, param_dics, args.multiplied_indices, args.summed_indices, 
                            alpha=args.alpha, n_restarts_optimizer=args.n_restarts_optimizer, 
                            normalize_y=args.normalize_y, random_state=args.gpr_random_state)
    
    info = collect_participant_info()
    print(f"Participant Info: {info}")

    # Create a window 
    win = visual.Window([1800, 1200], color='grey', units='pix', fullscr=False)
    line = visual.Line(win, start=(-500, 0), end=(500, 0), lineColor='black')
    line_leftend = visual.Line(win, start=(-500, -10), end=(-500, 10), lineColor='black')
    line_rightend = visual.Line(win, start=(500, -10), end=(500, 10), lineColor='black')
    left_label = visual.Rect(win, width=300, height=300, pos=(-500, -150), fillColor=None, lineColor='black')
    right_label = visual.Rect(win, width=300, height=300, pos=(500, -150), fillColor=None, lineColor='black')
    question_label = visual.Rect(win, width=300, height=300, pos=(0, 150), fillColor=None, lineColor='black')

    prompt = visual.TextStim(win, text='', pos=(0, -400), color='black')
    marker = visual.Line(win, start=(0, -10), end=(0, 10), lineColor='orange', lineWidth=3)

    # masking image
    img_stim = visual.ImageStim(
        win=win,
        image='./nlt_main/images/random_noise.png',
        pos=(0, 150),
        size=(300, 300)
    )

    visuals={'win':win,
            'line':line,
            'line_leftend':line_leftend,
            'line_rightend':line_rightend,
            'left_label':left_label,
            'right_label':right_label,
            'question_label':question_label,
            'prompt':prompt,
            'marker':marker,
            'img_stim':img_stim
            }
    
    run_experiment(args, gpr, visuals, info)
    
    