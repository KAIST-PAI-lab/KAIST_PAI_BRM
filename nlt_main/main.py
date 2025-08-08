from psychopy import visual, core, event, gui
from psychopy.visual import ElementArrayStim
from sklearn.gaussian_process.kernels import Kernel
import random
import os
import numpy as np
import pandas as pd
import time
import pickle
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')
import argparse

# Suppress Psychopy logging messages
from psychopy import logging
logging.console.setLevel(logging.ERROR)

# Import some functions from utils
from nlt_main.utils.utils import collect_participant_info, save_results
from nlt_main.NLE.draw_dots import draw_grid_position, calculate_dot_size
from gpal.gpal_optimize import gpal_optimize

def show_instructions(visuals, text):
    """
    Display the instructions for the number line estimation task.
    """
    win=visuals['win']
    line=visuals['line']
    line_leftend=visuals['line_leftend']
    line_rightend=visuals['line_rightend']
    left_label=visuals['left_label']
    right_label=visuals['right_label']
    

    positions = draw_grid_position(500, 4, 300, 300, box_center=(500, -200), padding=25)
    right_dots = ElementArrayStim(win,
                                   nElements=500,
                                   xys=positions,
                                   sizes=4,
                                   colors='orange',
                                   elementTex=None,
                                   elementMask='circle',
                                   interpolate=True)
    line.draw()
    line_leftend.draw()
    line_rightend.draw()
    left_label.draw()
    right_label.draw()
    right_dots.draw()


    instructions = visual.TextStim(win, text=text, color='black', wrapWidth=1500, pos=(0, -500), height=30)

    instructions.draw()
    win.flip()
    event.waitKeys(keyList=['space'])

def trial(number, dot_size, visuals, max_number=100):
    """
    Display a single trial of the number line estimation task.
    """

    win=visuals['win']
    line=visuals['line']
    line_leftend=visuals['line_leftend']
    line_rightend=visuals['line_rightend']
    left_label=visuals['left_label']
    right_label=visuals['right_label']
    question_label=visuals['question_label']
    prompt=visuals['prompt']
    marker=visuals['marker']
    img_stim=visuals['img_stim']


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
    clicked = False
    start_time = time.time()
    
    line.draw()
    line_leftend.draw()
    line_rightend.draw()
    left_label.draw()
    right_label.draw()
    question_label.draw()
    
    # 도트 표시
    right_dots.draw()
    question_dots.draw()
    win.flip()

    while not clicked:
        if time.time() - start_time > 2:
            img_stim.draw()
            question_label.draw()
            line.draw()
            line_leftend.draw()
            line_rightend.draw()
            left_label.draw()
            right_label.draw()
            right_dots.draw()
            win.flip()

        # 마우스 클릭 대기
        mouse = event.Mouse(win=win)
        if mouse.getPressed()[0]:
            x, y = mouse.getPos()
            timestep = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            end_time = time.time()
            reaction_time = end_time - start_time

            if x < -500 or x > 500 or y < -20 or y > 20:
                continue

            clicked = True
    
            # 클릭 위치 표시
            marker.pos = (x, 0)

            # image masking
            marker.draw()
            win.flip()

    # Get estimation
    est = round((x + 500) / 1000 * max_number, 2)
    res = [{'timestamp': timestep, 'given_number': number, 'upper_bound': max_number, 'estimation': est, 'estimation_rt': reaction_time, 'stimulus_ts_on': start_time, 'stimulus_ts_off': end_time, 'size_control': dot_size}]
    text = "계속 진행하려면 스페이스바를 누르세요."
    prompt = visual.TextStim(win, text=text, color='black', wrapWidth=1500, pos=(0, -500), height=30)

    # Draw the final screen with the estimation mark
    line.draw()
    line_leftend.draw()
    line_rightend.draw()
    left_label.draw()
    right_label.draw()
    prompt.draw()
    right_dots.draw()
    marker.draw()
    win.flip()

    return res

def run_NLE_gpal(gpr, records, tracks, block_len, block_idx, trial_idx, visuals, return_std=True, return_cov=False, size_control=False):
    
    block_start=block_len*block_idx
    
    max_number_candidates = [500]
    if trial_idx==0:
        max_number = max_number_candidates[random.randint(0, len(max_number_candidates)-1)]
        number = random.randint(5, max_number)
        pMean=0
        pStd=1
        lml=0
    else:
        mask=lambda X:X<=500
        dv1_spec=[5, 500, int((500-5)/5+1)]
        dv_spec=[dv1_spec]
        dvs=records[:-1, block_start:block_start+trial_idx]
        est=records[-1, block_start:block_start+trial_idx]
        result, pMean, pStd, lml = gpal_optimize(gpr=gpr, 
                                                 nDV=records.shape[0]-1,
                                                 dvs=dvs, 
                                                 est=est, 
                                                 dvSpecs=dv_spec, 
                                                 masking=mask)
        number=int(result[0].item())
        max_number=500

    dot_size = 4 if not size_control else calculate_dot_size(max_number_size=4, number=number, max_number=max_number)

    res = trial(number, dot_size, visuals, max_number=max_number)
    
    res[0]['size_control'] = 1 if size_control else 0
    records[0][block_start+trial_idx]=res[0]['given_number']
    records[1][block_start+trial_idx]=res[0]['estimation']
    tracks[0][block_start+trial_idx]=pMean
    tracks[1][block_start+trial_idx]=pStd
    res[0]['posterior_mean'] = np.max(pMean)
    res[0]['posterior_std'] = np.max(pStd)
    res[0]['log_marginal_likelihood'] = lml
    
    if trial_idx==0:
        constant_value=np.exp(gpr.kernel.k1.theta[0])
        length_scale=np.exp(gpr.kernel.k2.theta[0])
    else:
        constant_value=np.exp(gpr.kernel_.k1.theta[0])
        length_scale=np.exp(gpr.kernel_.k2.theta[0])
    res[0]['constant_value']=constant_value
    res[0]['length_scale'] = length_scale
    
    #block_res.extend(res)
    event.waitKeys(keyList=['space'])

    return res

def run_NLE_ado(engine, visuals, size_control=False):
    """
    Run a single trial of the number line estimation task using ADO optimization.
    """

    # Get the design using the ADO engine
    design = engine.get_design('optimal')
    dot_size = 4 if not size_control else calculate_dot_size(max_number_size=4, number=int(design['given_number']), max_number=500)

    res = trial(int(design['given_number']), dot_size, visuals, max_number=500)

    res[0]['size_control'] = 1 if size_control else 0
    
    res[0].update({f'post_mean_{k}': v for k, v in engine.post_mean.items()})
    res[0].update({f'post_sd_{k}': v for k, v in engine.post_sd.items()})


    # Update the engine with the design and response
    response = int(res[0]['estimation'])
    engine.update(design, response)

    event.waitKeys(keyList=['space'])

    return res

def run_NLE_random(visuals, given_number, size_control=False):
    """
    Run a single trial of the number line estimation task with balanced randomization.
    """
    dot_size = 4 if not size_control else calculate_dot_size(max_number_size=4, number=given_number, max_number=500)
    res = trial(given_number, dot_size, visuals, max_number=500)
    res[0]['size_control'] = 1 if size_control else 0
    event.waitKeys(keyList=['space'])
    return res


def run_gpal_block(gpr, visuals, n_trials, n_DVs, return_std=True, return_cov=False):
    """
    Run a block of trials using GPAL optimization.
    """
    block_len = n_trials * 2  # Two trials per block
    num_cands=int((500-5)/5+1)
    record_array=np.zeros((n_DVs+1, block_len))
    track_array=np.zeros((2, block_len, num_cands))
    block_res = []
    size_control_list = [True]*n_trials + [False]*n_trials  # Second block with variable dot size
    random.shuffle(size_control_list)  # Shuffle the order of blocks

    for idx, size_control_flag in enumerate(size_control_list):
        results=run_NLE_gpal(gpr, record_array, track_array, block_len, 0, idx, visuals, return_std, return_cov, size_control=size_control_flag)
        block_res.extend(results)

    return block_res, track_array 

def run_ado_block(engine, visuals, n_trials):
    """
    Run a block of trials using ADO optimization.
    """

    size_control_list = [True]*int(n_trials) + [False]*int(n_trials) 
    random.shuffle(size_control_list)  # Shuffle the order of blocks

    block_len = n_trials*2
    block_res = []
    
    for trial_idx in range(block_len):
        res = run_NLE_ado(engine, visuals, size_control_list[trial_idx])
        block_res.extend(res)

    return block_res

def run_random_block(visuals, n_trials):

    """
    Run a block of trials with balanced randomization.
    """
    block_len = n_trials*2
    block_res = []
    
    # Generate a list of given numbers
    given_number_list = np.linspace(5,480,20)
    random.shuffle(given_number_list)  # Shuffle the order of given numbers

    # Generate a list of size control flags
    size_control_list = [True]*int(n_trials) + [False]*int(n_trials) 
    random.shuffle(size_control_list)  # Shuffle the order of blocks

    for trial_idx in range(block_len):
        given_number = int(given_number_list[trial_idx])
        size_control = size_control_list[trial_idx]
        res = run_NLE_random(visuals, given_number, size_control)
        block_res.extend(res)

    return block_res


def run_experiment(config, gpr, engine, visuals, info):

    intro_text = "이제부터 선 위에 나타나는 박스의 위치를 추정하는 과제를 수행하게 됩니다. \n" \
                "선의 양 끝에는 두 개의 박스가 있습니다. \n" \
                "왼쪽 박스는 비어 있고, 오른쪽 박스에는 여러 개의 점이 들어 있습니다. \n" \
                "화면 위쪽에 나타나는 박스에 들어있는 점의 수를 보고, \n" \
                "이 점들이 어디쯤에 있다고 생각되는지, \n" \
                "선 위의 적절한 위치를 마우스로 클릭해 표시해주세요. \n" \
                "준비가 되면 스페이스바를 눌러 시작합니다."
    
    show_instructions(visuals, intro_text)

    num_trials = config.get('n_trials')
    n_DVs=config.get('n_DVs') 
    return_std=config.get('return_std')
    return_cov=config.get('return_cov')
    block_len=2*num_trials
    num_cands=int((500-5)/5+1)

    record_array_pre=np.zeros((n_DVs+1, 5))
    track_array_pre=np.zeros((2, 5, num_cands))
    # Run the pre-NLE block (test trials)
    for preIdx in range(5):
        results = run_NLE_gpal(gpr, record_array_pre, track_array_pre, block_len, 0, 0, visuals, return_std, return_cov) # for 5 test trials

    test_text = "연습이 종료되었습니다. \n 궁금한 점이 있으시면 질문해주세요."  
    show_instructions(visuals, test_text) 

    pid = int(info['Participant ID'])
    if pid % 3 == 0:
        # GPAL -> ADO -> Random
        print("Running GPAL block...")
        block_res, track_array = run_gpal_block(gpr, visuals, num_trials, n_DVs, return_std, return_cov)
        print("Running ADO block...")
        ado_result = run_ado_block(engine, visuals, num_trials)
        print("Running Random block...")
        random_result = run_random_block(visuals, num_trials)

    elif pid % 3 == 1:
        # ADO -> Random -> GPAL
        print("Running ADO block...")
        ado_result = run_ado_block(engine, visuals, num_trials)
        print("Running Random block...")
        random_result = run_random_block(visuals, num_trials)
        print("Running GPAL block...")
        block_res, track_array = run_gpal_block(gpr, visuals, num_trials, n_DVs, return_std, return_cov)

    else:
        # Random -> GPAL -> ADO
        print("Running Random block...")
        random_result = run_random_block(visuals, num_trials)
        print("Running GPAL block...")
        block_res, track_array = run_gpal_block(gpr, visuals, num_trials, n_DVs, return_std, return_cov)
        print("Running ADO block...")
        ado_result = run_ado_block(engine, visuals, num_trials)

    
    save_results_dir=config.get('save_results_dir')
    save_results_dir = os.path.join(save_results_dir, f"participant_{info['Participant ID']}")
    if not os.path.exists(save_results_dir):
        os.makedirs(save_results_dir, exist_ok=True)

    
    # Save the results to a CSV file
    filename = f"ado_results_{info['Participant ID']}.csv"
    save_results(save_results_dir, ado_result, info, filename=filename)

    # Save the results to a CSV file
    filename = f"gpal_results_{info['Participant ID']}.csv"
    save_results(save_results_dir, block_res, info, filename=filename)

    filename = f"random_results_{info['Participant ID']}.csv"
    save_results(save_results_dir, random_result, info, filename=filename)

    filename = f"gpal_posterior_mean_{info['Participant ID']}.csv"
    np.savetxt(os.path.join(save_results_dir, filename), track_array[0], delimiter=',')
    filename = f"gpal_posterior_std_{info['Participant ID']}.csv"
    np.savetxt(os.path.join(save_results_dir, filename), track_array[1], delimiter=',')


    sbj=f"{info['Participant ID']}"
    save_models_dir=config.get('save_models_dir')
    save_models_sbj_dir=save_models_dir+'/'+sbj  # The directory to save the optimized model for each subject

    if not os.path.exists(save_models_sbj_dir):
        os.mkdir(save_models_sbj_dir)
    filepath=f"{save_models_sbj_dir}/GPR_{info['Participant ID']}.pkl"
    with open(filepath, "wb") as f:
        pickle.dump(gpr, f)
    

    print("Experiment completed. Results saved.")

    win=visuals['win']
    win.flip()

    # Display Ending Message
    text = "실험이 종료되었습니다. \n 참여해주셔서 감사합니다! \n 스페이스바를 눌러 종료하세요."
    prompt = visual.TextStim(win, text=text, color='black', wrapWidth=1500, pos=(0, 0), height=30)
    prompt.draw()
    win.flip()
    event.waitKeys(keyList=['space'])
    win.close()
    core.quit()
    
    '''
    random.seed(args.seed)  # For reproducibility

    # Collect participant information
    info = collect_participant_info()
    print(f"Participant Info: {info}")

    # Create a window
    win = visual.Window([1800, 1200], color='grey', units='pix', fullscr=True)
    line = visual.Line(win, start=(-500, 0), end=(500, 0), lineColor='black')
    line_leftend = visual.Line(win, start=(-500, -10), end=(-500, 10), lineColor='black')
    line_rightend = visual.Line(win, start=(500, -10), end=(500, 10), lineColor='black')
    left_label = visual.Rect(win, width=300, height=250, pos=(-500, -150), fillColor=None, lineColor='black')
    right_label = visual.Rect(win, width=300, height=250, pos=(500, -150), fillColor=None, lineColor='black')
    question_label = visual.Rect(win, width=300, height=250, pos=(0, 150), fillColor=None, lineColor='black')

    prompt = visual.TextStim(win, text='', pos=(0, -400), color='black')
    marker = visual.Line(win, start=(0, -10), end=(0, 10), lineColor='orange', lineWidth=3)

    # masking image
    img_stim = visual.ImageStim(
        win=win,
        image='./images/random_noise.png',
        pos=(0, 150),
        size=(300, 250)
    )
    show_instructions()

    # Run the pre-NLE block (test trials)
    results = run_NLE_block(5) # for 5 test trials

    # Run the NLE block
    num_trials = args.n_trials 

    size_control_list = [True]*num_trials + [False]*num_trials  # Second block with variable dot size
    random.shuffle(size_control_list)  # Shuffle the order of blocks

    for size_control_flag in size_control_list:
        if size_control_flag:
            results.extend(run_NLE_block(1, size_control=True))  # Second block with variable dot size
        else:
            results.extend(run_NLE_block(1))

    # Save the results to a CSV file
    filename = f"results_{info['Participant ID']}_{info['Name']}.csv"
    save_results(results, info, filename=filename)
    print("Experiment completed. Results saved.")

    win.close()
    core.quit()
    '''