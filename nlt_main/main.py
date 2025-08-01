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
from gpal.gpal_optimize import gpal_optimize2D

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


    instructions = visual.TextStim(win, text=text, color='black', wrapWidth=1500, pos=(0, -500))

    instructions.draw()
    win.flip()
    event.waitKeys(keyList=['space'])

def trial(number, dot_size, visuals, max_number=100):
    """
    Run a single trial of the number line estimation task.
    All dots have same size, so this function is used first half of the number-line task.
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
    prompt.text = "계속 진행하려면 스페이스바를 누르세요."

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


def run_NLE_block(gpr, records, trial_idx, visuals, return_std=True, return_cov=False, size_control=False):
    """
    AFTER GPAL has been implemented, this function will run a block of the number line estimation task, using GPAL to optimize the given-number and upperbound number.

    if i == 0:
        num, max_num = random.sample(range(1, 101), 2) # 임의의 숫자 두개
    else:
        num, max_num = GPAL.GPAL(results[-1]['response'], results[-1]['max_number']) # 가장 최근 응답과 상한선을 기반으로 숫자 선택
    
    Run a block of the number line estimation task.
    The first block uses a fixed dot size, while the second block uses variable dot sizes. 
    """

    #for i in range(num_trials):
    max_number_candidates = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    if trial_idx==0:
        max_number = random.choice(max_number_candidates)
        number = random.randint(5, max_number)  # 5이상 상한선 미만 임의의 숫자 선택 (would be changed later)
        pMean=0
        pStd=1
        lml=0
    else:
        mask=lambda X,Y: X<Y
        dv1_spec=[5, 500, int((500-5)/1+1)]
        dv2_spec=[50, 500, int((500-50)/50+1)]
        dv1=records[0][:trial_idx]
        dv2=records[1][:trial_idx]
        est=records[2][:trial_idx]
        result, pMean, pStd, lml = gpal_optimize2D(gpr, dv1, dv2, est, dv1_spec, dv2_spec, mask, return_std, return_cov)
        number=int(result[0].item())
        max_number=int(result[1].item())

    dot_size = 4 if not size_control else calculate_dot_size(max_number_size=4, number=number, max_number=max_number)
    #print(f"number: {number}")
    # print(f"max_number: {max_number}")
    # print(f"dot_size: {dot_size}")
    res = trial(number, dot_size, visuals, max_number=max_number)
    
    res[0]['size_control'] = 1 if size_control else 0
    records[0][trial_idx]=res[0]['given_number']
    records[1][trial_idx]=res[0]['upper_bound']
    records[2][trial_idx]=res[0]['estimation']
    res[0]['posterior_mean'] = pMean
    res[0]['posterior_std'] = pStd
    res[0]['log_marginal_likelihood'] = lml
    #block_res.extend(res)
    event.waitKeys(keyList=['space'])

    return res


def run_experiment(args, gpr, visuals, info):

    intro_text = "본 실험에서, 이 작업에는 다음과 같은 숫자 선이 있습니다.\n" \
                "각 숫자 선의 왼쪽 끝에는 점이 없고 오른쪽 쪽 끝에는 몇 개의 점이 있습니다.\n" \
                "선 위에 점이 잠시 나타납니다.\n" \
                "선 위에서 점이 위치할 곳을 클릭해 주세요.\n" \
                "스페이스바를 눌러 시작하세요."

    show_instructions(visuals, intro_text)

    num_trials = args.n_trials 
    record_array_pre=np.zeros((3, 5))


    # Run the pre-NLE block (test trials)
    for preIdx in range(5):
        results = run_NLE_block(gpr, record_array_pre, 0, visuals, return_std=args.return_std, return_cov=args.return_cov) # for 5 test trials

    test_text = "연습이 종료되었습니다. \n 궁금한 점이 있으시면 질문해주세요. \n 계속 진행하시려면 스페이스바를 누르세요."  
    show_instructions(visuals, test_text)  

    # Run the NLE block
    block_res = []
    record_array=np.zeros((3, num_trials*2))
    size_control_list = [True]*num_trials + [False]*num_trials  # Second block with variable dot size
    random.shuffle(size_control_list)  # Shuffle the order of blocks

    for idx, size_control_flag in enumerate(size_control_list):
        if size_control_flag:
            results=run_NLE_block(gpr, record_array, idx, visuals, args.return_std, args.return_cov, size_control=True)  # Second block with variable dot size
        else:
            results=run_NLE_block(gpr, record_array, idx, visuals, args.return_std, args.return_cov)
        block_res.extend(results) 

    # Save the results to a CSV file
    filename = f"results_{info['Participant ID']}_{info['Name']}.csv"
    save_results(args.save_results_dir, block_res, info, filename=filename)

    sbj=args.subject_prefix+f"_{info['Participant ID']}_{info['Name']}"
    save_models_sbj_dir=args.save_models_dir+'/'+sbj  # The directory to save the optimized model for each subject

    if not os.path.exists(save_models_sbj_dir):
        os.mkdir(save_models_sbj_dir)
    filepath=f"{save_models_sbj_dir}/GPR_{info['Participant ID']}_{info['Name']}.pkl"
    with open(filepath, "wb") as f:
        pickle.dump(gpr, f)
    

    print("Experiment completed. Results saved.")

    win=visuals['win']
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