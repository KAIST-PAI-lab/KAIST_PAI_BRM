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
from utils.utils import collect_participant_info, save_results
from NLE.draw_dots import draw_grid_position, calculate_dot_size
from nlt.gpal.gpal_optimize import gpal_optimize

def show_instructions():
    """
    Display the instructions for the number line estimation task.
    """
    positions = draw_grid_position(500, 4, 300, 250, box_center=(500, -150), padding=20)
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


    instructions = visual.TextStim(win, text="In this task, there will be a number line like this one.\n"
                                             "Each number line will have no dots at one end and some dots at the other end.\n"
                                             "There will be a number of dots appearing for a short time above a line.\n"
                                             "Please click on the line where the dot goes.\n"
                                             "Press the space bar to begin.", color='black', wrapWidth=1500, pos=(0, -400))

    instructions.draw()
    win.flip()
    event.waitKeys(keyList=['space'])

def trial(number, dot_size, max_number=100):
    """
    Run a single trial of the number line estimation task.
    All dots have same size, so this function is used first half of the number-line task.
    """

    right_dots_pos = draw_grid_position(max_number, 4, 300, 250, box_center=(500, -150), padding=20)
    right_dots = ElementArrayStim(win,
                                   nElements=max_number,
                                   xys=right_dots_pos,
                                   sizes=4,
                                   colors='orange',
                                   elementTex=None,
                                   elementMask='circle',
                                   interpolate=True)

    question_dots_pos = draw_grid_position(number, dot_size, 300, 250, box_center=(0, 150), padding=20)
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

            if x < -500 or x > 500:
                continue

            clicked = True
    
            # 클릭 위치 표시
            marker.pos = (x, 0)

            # image masking
            marker.draw()
            win.flip()

    # Get estimation
    est = round((x + 500) / 1000 * max_number, 2)
    res = [{'timestamp': timestep, 'given_number': number, 'upperbound': max_number, 'estimation': est, 'estimation_rt': reaction_time, 'stimulus_ts_on': start_time, 'stimulus_ts_off': end_time, 'size_control': dot_size}]
    prompt.text = "Press spacebar to continue."

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


def run_NLE_block(gpr, records, num_trials, return_std=True, return_cov=False, size_control=False):
    """
    AFTER GPAL has been implemented, this function will run a block of the number line estimation task, using GPAL to optimize the given-number and upperbound number.

    if i == 0:
        num, max_num = random.sample(range(1, 101), 2) # 임의의 숫자 두개
    else:
        num, max_num = GPAL.GPAL(results[-1]['response'], results[-1]['max_number']) # 가장 최근 응답과 상한선을 기반으로 숫자 선택
    
    Run a block of the number line estimation task.
    The first block uses a fixed dot size, while the second block uses variable dot sizes. 
    """

    block_res = []

    for i in range(num_trials):
        max_number_candidates = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
        if i==0:
            max_number = random.choice(max_number_candidates)
            number = random.randint(5, max_number)  # 5이상 상한선 미만 임의의 숫자 선택 (would be changed later)
        else:
            number, max_number, pMean, pStd, lml = gpal_optimize(gpr, records[0], records[1], records[2], return_std, return_cov)
        
        dot_size = 4 if not size_control else calculate_dot_size(max_number_size=4, number=number, max_number=max_number)
        res = trial(number, dot_size, max_number=max_number)
        
        res[0]['size_control'] = 1 if size_control else 0
        records[0][i]=res[0]['given_number']
        records[1][i]=res[0]['upperbound']
        records[2][i]=res[0]['estimation']
        res[0]['posterior_mean'] = pMean
        res[0]['posterior_std'] = pStd
        res[0]['log_marginal_likelihood'] = lml
        block_res.extend(res)
        event.waitKeys(keyList=['space'])

    return block_res

def run_experiment(args, gpr, subject_id):

    #parser = argparse.ArgumentParser(description='Run the Number Line Estimation Experiment')
    random.seed(args.seed)
    
    sbj=args.subject_prefix+str(subject_id)
    save_models_sbj_dir=args.save_models_dir+'/'+sbj  # The directory to save the optimized model for each subject
    os.makedirs(save_models_sbj_dir, exist_ok=True)
    
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
        image='./nlt_main/images/random_noise.png',
        pos=(0, 150),
        size=(300, 250)
    )
    show_instructions()

    ## record_array[0][i] : Given number in i-th trial.
    ## record_array[1][i] : Upper bound in i-th trial.
    ## record_array[2][i] : Given number estimate in i-th trial.
    num_trials = args.n_trials 
    record_array=np.zeros((3, num_trials))


    # Run the pre-NLE block (test trials)
    results = run_NLE_block(gpr, record_array, 5, return_std=args.return_std, return_cov=args.return_cov) # for 5 test trials

    # Run the NLE block
    size_control_list = [True]*num_trials + [False]*num_trials  # Second block with variable dot size
    random.shuffle(size_control_list)  # Shuffle the order of blocks

    for size_control_flag in size_control_list:
        if size_control_flag:
            results.extend(run_NLE_block(gpr, record_array, 1, args.return_std, args.return_cov, size_control=True))  # Second block with variable dot size
        else:
            results.extend(run_NLE_block(gpr, record_array, 1, args.return_std, args.return_cov))

    # Save the results to a CSV file
    filename = f"results_{info['Participant ID']}_{info['Name']}.csv"
    save_results(args.save_results_dir, results, info, filename=filename)

    filepath=os.path.join(save_models_sbj_dir, f'/GPR_{info['Participant ID']}_{info['Name']}_{info['timestamp']}.pkl')
    with open(filepath, "wb") as f:
        pickle.dump(gpr, f)

    print("Experiment completed. Results saved.")

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