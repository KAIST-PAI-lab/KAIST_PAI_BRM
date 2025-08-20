from psychopy import event, logging
import numpy as np
import pandas as pd
import os, yaml, random, warnings

# Import some functions from utils
from gpal.gpal_optimize import gpal_optimize
from gpal.gpr_instance import GPRInstance
from gpal.utils import argsConstructor

# Import number-line estimation task setup functions
from nlt_setup import show_and_get_response, initialize_psychopy

# Suppress warning messages
warnings.filterwarnings('ignore')
logging.console.setLevel(logging.ERROR)

# Initialize PsychoPy visuals
# Default fullscreen mode is False
# initialize_psychopy function is called to set up the PsychoPy visual elements 
visuals = initialize_psychopy(fullscr=False)

# Load Experiment Configuration File
try:
    with open('gpal/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    print("Configuration loaded.")

except FileNotFoundError:
    print("config.yaml file cannot be found.")

## Arguments related to the experimental environment
n_trials = config.get('n_trials')                             # Number of experiment trials for a subject.
seed = config.get('seed')                                     # A random seed value for reproducibility.
n_DVs = config.get('n_DVs')                                   # The number of design variables subject to optimization.

## Arguments related to initializing the GPR instance.
n_kernels = config.get('n_kernels')                           # The number of individual kernels to be combined. Should be a positive integer.
alpha = config.get('alpha')                                   # A value added to the diagonal of the kernel matrix during fitting.
normalize_y = config.get('normalize_y')                       # A binary mask indicating whether to normalize the target values while fitting.
n_restarts_optimizer = config.get('n_restarts_optimizer')     # The number of restarts of the optimizer to find the optimal kernel parameters.
type_kernels_index = config.get('type_kernels_index')         # A list of indices of kernels to be combined. Refer to 'kernelTypeDic' above.
parameters_list = config.get('parameters_list')               # A list of list of arguments to be fed to each kernel.
format = config.get('format')                                 # A string indicating how the kernels should be combined.
gpr_random_state = config.get('gpr_random_state')             # A parameter determining random number generation in initializing the centers of the GP regressor.

## Arguments related to optimizing the GPR instance.
return_std = config.get('return_std')                         # A binary mask indicating whether to return standard deviation of posterior distribution at each query value.
return_cov = config.get('return_cov')                         # A binary mask indicating whether to return covaraince matrix of posterior distribution at each query value.

## Arguments related to running an experiment.
save_results_dir = config.get('save_results_dir')             # A directory to store the task results.
save_models_dir = config.get('save_models_dir')               # A directory to store the trained Gaussian process regressor models.

## Construct kernel type specifications and parameter dictionaries for initializing GPRInstance
kernel_type, kernel_param = argsConstructor(N=n_kernels, 
                                            kernelTypeIdx=type_kernels_index, 
                                            paramsList=parameters_list)

## Initializing GPRInstance
kernel, gpr = GPRInstance(typeKernel=type_kernels, 
                            paramKernel=param_dics, 
                            format=format, 
                            alpha=alpha, 
                            n_restarts_optimizer=n_restarts_optimizer, 
                            normalize_y=normalize_y, 
                            random_state=gpr_random_state)


# Determine the order of size controlling of trials
# It randomly samples half of the trials to be size-controlled.
size_control_order = random.sample([True] * (n_trials//2) + [False] * (n_trials//2), n_trials)

# Initialize record array.
# In 1D number-line estimation task, 
# record_array should store the optimized design value in the first row
# and the participant's estimation in the second row.
# record_array is updated after each trial.
record_array = np.zeros((n_DVs+1, n_trials))    

# Define upperbound value of 1D number-line estimation task
max_number = 500

# Run trials with GPAL
for trial_idx in range(n_trials):

    # If this is the first trial, initialize parameters
    if trial_idx == 0:
        given_number = 5 * random.randint(1, 100)
        pMean = 0
        pStd = 1
        lml = 0

    # If this is not the first trial, use previous results to optimize current trial
    else:
        mask = lambda X: X <= 500                                                    # Masking function to filter out values greater than 500
        dv1_spec = [5, 500, 5]                                                       # Design variable 1 specification: [start, end, interval]
        dv_spec = [dv1_spec]                                                         # List of all design variable specifications. In this case, there is only one design variable.
        dvs = record_array[:-1, :trial_idx]                                          # Past design variable values shown to participant
        est = record_array[-1, :trial_idx]                                           # Past participant responses
        result, pMean, pStd, lml = gpal_optimize(gpr=gpr, 
                                                 nDV=n_DVs,                          # Number of design variables to be optimized
                                                 dvs=dvs, 
                                                 est=est, 
                                                 dvSpecs=dv_spec, 
                                                 masking=mask)                          
        given_number = int(result[0].item())                                         # Optimized design variable value

    # Show the dots and get response from participant
    # If size_control is enabled, adjust the dot size, else use default size
    response = show_and_get_response(given_number, visuals, max_number=max_number, size_control=size_control_order[trial_idx])

    # Record the results for the next trial
    # Row 0: given number 
    # Row 1: response of participant
    record_array[0][trial_idx] = given_number
    record_array[1][trial_idx] = response

    # Wait for space key to continue to the next trial
    event.waitKeys(keyList=['space'])  

# Save experiment results to CSV file
results_df = pd.DataFrame(record_array.T, columns=['given_number', 'response'])
results_df.to_csv(os.path.join(save_results_dir, f'results_trial_{n_trials}.csv'), index=False)

# Close the PsychoPy window
visuals['win'].close()  