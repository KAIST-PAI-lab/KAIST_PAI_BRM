from psychopy import visual, event, logging
from psychopy.visual import ElementArrayStim
from sklearn.gaussian_process.kernels import Kernel
import numpy as np
import pandas as pd
import math, time, os, yaml, random, warnings

# Import some functions from utils
from gpalexp.gpal_optimize import gpal_optimize
from gpalexp.gpr_instance import GPRInstance
from gpalexp.utils import argsConstructor

# Suppress warning messages
warnings.filterwarnings('ignore')
logging.console.setLevel(logging.ERROR)

def draw_grid_position(n_dots, dot_size, boxW, boxH, box_center=(0, 0), padding=10):
    """
    Draws a grid of positions for the dots within a specified box, 
    so that the dots are evenly spaced.

    Args:
    - n_dots: number of dots
    - dot_size: size(diameter) of dots
    - boxW, boxH: dimensions of the box (width, height)
    - box_center: center position of the box (x, y)
    - padding: padding between dots and box edges

    Returns:
    - jittered_positions: list of tuples representing the (x, y) positions of the dots
    """
    itemWH = dot_size * 1.75
    usable_boxW = boxW - 2 * (padding + dot_size / 2)
    usable_boxH = boxH - 2 * (padding + dot_size / 2)

    X = int(usable_boxW // itemWH)
    Y = int(usable_boxH // itemWH)

    if X * Y < n_dots:
        raise ValueError("Not enough grid positions for the number of dots.")
    
    grid_x = np.linspace(-usable_boxW / 2, usable_boxW / 2, X)
    grid_y = np.linspace(-usable_boxH / 2, usable_boxH / 2, Y)
    grid_X, grid_Y = np.meshgrid(grid_x, grid_y)
    grid_positions = list(zip(grid_X.flatten(), grid_Y.flatten()))
    random.shuffle(grid_positions)

    selected_positions = grid_positions[:n_dots]

    jittered_positions = []
    jitter_ratio = 0.3
    for x, y in selected_positions:
        jitter_x = random.uniform(-itemWH * jitter_ratio, itemWH * jitter_ratio)
        jitter_y = random.uniform(-itemWH * jitter_ratio, itemWH * jitter_ratio)
        final_x = x + jitter_x + box_center[0]
        final_y = y + jitter_y + box_center[1]
        jittered_positions.append((final_x, final_y))

    return jittered_positions

def calculate_dot_size(max_number_size, number, max_number=100):
    """
    Calculate the size of the dots based on the number of dots.
    Size of the given number of dots is calculated to be same as the cumulative area of the upperbound dots.

    Args:
    - max_number_size: maximum size(diameter) of the dots
    - number: number of dots
    - max_number: maximum number of dots

    Returns:
    - dot_size: size(diameter) of the dots
    """
    original_r = max_number_size / 2
    total_upper_area = math.pi * (original_r ** 2) * max_number

    current_dot_area = total_upper_area / number
    current_r = math.sqrt(current_dot_area / math.pi)

    stimdot_size1 = math.floor(current_r * 2)
    stimdot_size2 = math.ceil(current_r * 2)

    error1 = abs(total_upper_area - (stimdot_size1 / 2) ** 2 * math.pi * number)
    error2 = abs(total_upper_area - (stimdot_size2 / 2) ** 2 * math.pi * number)

    stimdot_size = stimdot_size1 if error1 < error2 else stimdot_size2
    return stimdot_size

def draw_base_components(visuals):
    """
    Draw base components (line, left_box, right_box, leftend, rightend, question_box) on win.

    Args:
    - win: PsychoPy window object
    - visuals: Dictionary of PsychoPy visual components
    """

    visuals['left_box'].draw()
    visuals['right_box'].draw()
    visuals['question_box'].draw()
    visuals['line'].draw()
    visuals['line_leftend'].draw()
    visuals['line_rightend'].draw()

def show_and_get_response(number, dot_size, visuals, max_number=500):
    """
    Show and get response from the user.

    Args:
    - number: number of dots
    - dot_size: size(diameter) of dots
    - visuals: dictionary of PsychoPy visual components to show
    - max_number: maximum number of dots

    Returns:
    - response: the user's response
    """
    win = visuals['win']
    marker = visuals['marker']
    img_stim = visuals['img_stim']

    # Draw dots & base visual components
    right_dots_pos = draw_grid_position(max_number, 4, 300, 300, box_center=(500, -200), padding=25)
    right_dots = ElementArrayStim(win,
                                   nElements=max_number,
                                   xys=right_dots_pos,
                                   sizes=4,
                                   colors='orange',
                                   elementTex=None,
                                   elementMask='circle',
                                   interpolate=True)

    question_dots_pos = draw_grid_position(number, dot_size, 300, 300, box_center=(0, 200), padding=25)
    question_dots = ElementArrayStim(win,
                                      nElements=number,
                                      xys=question_dots_pos,
                                      sizes=dot_size,
                                      colors='orange',
                                      elementTex=None,
                                      elementMask='circle',
                                      interpolate=True)
    draw_base_components(visuals)
    right_dots.draw()
    question_dots.draw()
    win.flip()
    start_time = time.time()

    # Defining mouse click flag
    clicked = False
    
    while not clicked:
        if time.time() - start_time > 2:
            # After 2 seconds, mask the given dots box
            img_stim.draw()
            right_dots.draw()
            draw_base_components(visuals)
            win.flip()

        # Waiting for mouse click
        mouse = event.Mouse(win=win)
        if mouse.getPressed()[0]:
            x, y = mouse.getPos()

            # If the click is outside the valid area, ignore it
            if x < -500 or x > 500 or y < -20 or y > 20:
                continue

            # If the click is valid, record the position and show the marker
            clicked = True
            marker.pos = (x, 0)
            marker.draw()
            win.flip()

    # Draw the response position
    visuals['line'].draw()
    visuals['line_leftend'].draw()
    visuals['line_rightend'].draw()
    visuals['left_box'].draw()
    visuals['right_box'].draw()
    right_dots.draw()
    marker.draw()
    win.flip()

    # Record the response
    response = {
        'given_number': number,
        'upper_bound': max_number,
        'estimation': int((x + 500) / 1000 * max_number),
    }

    return response

def gpal_trial(gpr, records, trial_idx, visuals, size_control=False):
    """
    Run a GPAL trial.

    Args:
    - gpr: Gaussian Process Regression model
    - records: historical data for the GPAL process (including design variables and responses)
    - trial_idx: index of the current trial
    - visuals: list of visual components for the trial
    - size_control: whether to control the size of the dots

    Returns:
    - res: response from the user
    """
    max_number = 500

    # If this is the first trial, initialize parameters
    if trial_idx == 0:
        number = 5 * random.randint(1, 100)
        pMean = 0
        pStd = 1
        lml = 0

    # If this is not the first trial, use previous results to optimize current trial
    else:
        mask = lambda X: X <= 500
        dv1_spec = [5, 500, 5]                                                  # Design variable 1 specification: (start, end, interval)
        dv_spec = [dv1_spec]                                                    # List of all design variable specifications
        dvs = records[:-1, :trial_idx]                                          # Past design variable values shown to participant
        est = records[-1, :trial_idx]                                           # Past participant responses
        result, pMean, pStd, lml = gpal_optimize(gpr=gpr, 
                                                 nDV=records.shape[0]-1,        # Number of design variables to be optimized
                                                 dvs=dvs, 
                                                 est=est, 
                                                 dvSpecs=dv_spec, 
                                                 masking=mask)
        number = int(result[0].item())                                          # Optimized design variable value

    # If size_control is enabled, adjust the dot size, else use default size
    dot_size = 4 if not size_control else calculate_dot_size(max_number_size=4, number=number, max_number=max_number)

    # Show the dots and get response from participant
    res = show_and_get_response(number, dot_size, visuals, max_number=max_number)

    # Record the results for the next trial
    records[0][trial_idx] = res['given_number']
    records[1][trial_idx] = res['estimation']

    return res


if __name__ == "__main__":

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
    type_kernels, param_dics = argsConstructor(N=n_kernels, 
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
    size_control_order = random.sample([True] * (n_trials//2) + [False] * (n_trials//2), n_trials)
    
    # Initialize record array
    record_array = np.zeros((n_DVs+1, n_trials))

    # Initialize Psychopy visual components
    win = visual.Window([1800, 1200], color='grey', units='pix', fullscr=False)
    line = visual.Line(win, start=(-500, 0), end=(500, 0), lineColor='black')
    line_leftend = visual.Line(win, start=(-500, -10), end=(-500, 10), lineColor='black')
    line_rightend = visual.Line(win, start=(500, -10), end=(500, 10), lineColor='black')
    left_box = visual.Rect(win, width=300, height=300, pos=(-500, -200), fillColor=None, lineColor='black')
    right_box = visual.Rect(win, width=300, height=300, pos=(500, -200), fillColor=None, lineColor='black')
    question_box = visual.Rect(win, width=300, height=300, pos=(0, 200), fillColor=None, lineColor='black')
    marker = visual.Line(win, start=(0, -10), end=(0, 10), lineColor='orange', lineWidth=3)
    img_stim = visual.ImageStim(
        win=win,
        image='./nlt_main/images/random_noise.png',
        pos=(0, 200),
        size=(300, 300)
    )

    visuals = {'win': win,
               'line': line,
               'line_leftend': line_leftend,
               'line_rightend': line_rightend,
               'left_box': left_box,
               'right_box': right_box,
               'question_box': question_box,
               'marker': marker,
               'img_stim': img_stim
               }        

    # Run trials with GPAL
    for trial_idx in range(n_trials):
        res = gpal_trial(gpr, record_array, trial_idx, visuals, size_control=size_control_order[trial_idx])
        event.waitKeys(keyList=['space'])  # Wait for space key to continue to the next trial

    # Save results to CSV file
    results_df = pd.DataFrame(record_array.T, columns=['given_number', 'estimation'])
    results_df.to_csv(os.path.join(save_results_dir, f'results_trial_{n_trials}.csv'), index=False)