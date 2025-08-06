import os
import argparse
import numpy as np
import warnings
import random
import pandas as pd
import pickle
import yaml
from datetime import datetime
from sklearn.gaussian_process import GaussianProcessRegressor


from psychopy import visual, core, event, gui
from psychopy.visual import ElementArrayStim
# Suppress Psychopy logging messages
from psychopy import logging
logging.console.setLevel(logging.ERROR)


from gpal.gpr_instance import *
from gpal.utils import argsConstructor
from nlt_main.utils.utils import collect_participant_info
#from nlt_main.NLE.draw_dots import draw_grid_position, calculate_dot_size
from nlt_main.main import run_experiment

os.environ["CUDA_VISIBLE_DEVICES"] = '2'  # Specifying the GPU to use


 

if __name__=="__main__":

    try:
        with open('gpal/config.yaml', 'r') as file:
            config=yaml.safe_load(file)
        
        print("Configuration loaded.")

    except FileNotFoundError:
        print("config.yaml file cannot be found.")
    
    
    ## Arguments related to the experimental environment
    n_trials=config.get('n_trials')                             # Number of experiment trials for a subject.
    seed=config.get('seed')                                     # A random seed value for reproducibility.
    n_DVs=config.get('n_DVs')                                   # The number of design variables subject to optimization.

    
    ## Arguments related to initializing the GPR instance.
    n_kernels=config.get('n_kernels')                           # The number of individual kernels to be combined. Should be a positive integer.
    alpha=config.get('alpha')                                   # A value added to the diagonal of the kernel matrix during fitting.
    normalize_y=config.get('normalize_y')                       # A binary mask indicating whether to normalize the target values while fitting.
    n_restarts_optimizer=config.get('n_restarts_optimizer')     # The number of restarts of the optimizer to find the optimal kernel parameters.
    type_kernels_index=config.get('type_kernels_index')         # A list of indices of kernels to be combined. Refer to 'kernelTypeDic' above.
    parameters_list=config.get('parameters_list')               # A list of list of arguments to be fed to each kernel.
    multiplied_indices=config.get('multiplied_indices')         # A list of lists indicating the kernels to be multiplied.
    summed_indices=config.get('summed_indices')                 # A list of lists indicating the kernels to be summed.
    gpr_random_state=config.get('gpr_random_state')             # A parameter determining random number generation in initializing the centers of the GP regressor.
    
    
    ## Arguments related to optimizing the GPR instance.
    return_std=config.get('return_std')                         # A binary mask indicating whether to return standard deviation of posterior distribution at each query value.
    return_cov=config.get('return_cov')                         # A binary mask indicating whether to return covaraince matrix of posterior distribution at each query value.
    
    
    ## Arguments related to running an experiment.
    save_results_dir=config.get('save_results_dir')             # A directory to store the task results.
    save_models_dir=config.get('save_models_dir')               # A directory to store the trained Gaussian process regressor models.


    

    warnings.filterwarnings("ignore")
    if seed is not None:
        random.seed(seed)
    
    
    type_kernels, param_dics=argsConstructor(N=n_kernels, 
                                             kernelTypeIdx=type_kernels_index, 
                                             paramsList=parameters_list)
    kernel, gpr=GPRInstance(nKernel=n_kernels, typeKernel=type_kernels, paramKernel=param_dics, 
                            mulIdx=multiplied_indices, sumIdx=summed_indices, 
                            alpha=alpha, n_restarts_optimizer=n_restarts_optimizer, 
                            normalize_y=normalize_y, random_state=gpr_random_state)
    print(f"Hyperparameters: {kernel.get_params()}")
    info = collect_participant_info()
    print(f"Participant Info: {info}")

    # Create a window 
    win = visual.Window([1800, 1200], color='grey', units='pix', fullscr=False)
    line = visual.Line(win, start=(-500, 0), end=(500, 0), lineColor='black')
    line_leftend = visual.Line(win, start=(-500, -10), end=(-500, 10), lineColor='black')
    line_rightend = visual.Line(win, start=(500, -10), end=(500, 10), lineColor='black')
    left_label = visual.Rect(win, width=300, height=300, pos=(-500, -200), fillColor=None, lineColor='black')
    right_label = visual.Rect(win, width=300, height=300, pos=(500, -200), fillColor=None, lineColor='black')
    question_label = visual.Rect(win, width=300, height=300, pos=(0, 200), fillColor=None, lineColor='black')

    prompt = visual.TextStim(win, text='', pos=(0, -400), color='black')
    marker = visual.Line(win, start=(0, -10), end=(0, 10), lineColor='orange', lineWidth=3)

    # masking image
    img_stim = visual.ImageStim(
        win=win,
        image='./nlt_main/images/random_noise.png',
        pos=(0, 200),
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
    
    run_experiment(config, gpr, visuals, info)
    
    