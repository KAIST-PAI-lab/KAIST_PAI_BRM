from psychopy import visual
import random
import pandas as pd
import numpy as np
import math

def draw_grid_position(n_dots, dot_size, boxW, boxH, box_center=(0, 0), padding=10):
    """
    Draws a grid of positions for the dots within a specified box, 
    so that the dots are evenly spaced.
    """
    itemWH = dot_size * 2
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
    Used in the second half of the number-line task.
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