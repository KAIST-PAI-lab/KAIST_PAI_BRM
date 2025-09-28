# Task details

# Comparison across blocks
# ~Overlay GP-inferred functions (mean function with credible region) for the GPAL and BR~
# ~Order: ADO, GPAL, BR~
# ~Figure title (ADO, GPAL, and BR)~
# ~Modify the font sizes as in other figures~
# ~Larger half-transperant dots to make repeated sampling at certain locations more visible~

# Target subjects
# Subject #25081110, Logarithmic funcs.
# Subject #25081116, Linear funcs.

#%%
import os
import random
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import psutil
import yaml
from gpalexp import *
from matplotlib.ticker import MultipleLocator
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

from simulation_NLE import psychometric_functions as psy_funcs
from utils_simulation import *

exp_data = get_results_data_points(data_dir="experiment results")

subject_code = "25081311" # 1213 and 1311


# load data points from the target subject

# cog models declaration
MLLM = getattr(psy_funcs, "mixed_log_linear")
OneCPM = getattr(psy_funcs, "one_cyclic_power")

font_size_title = 25
font_size_legend = 10
font_size_axis_labels = 15
font_size_tick_size = 10
data_points_size = 50
data_points_opacity = 0.4
legend_location = "lower right"

white_kernel_noise_bounds = (0.5, 1e5) # 기존 0.05

kernel = C(1.0)*RBF(length_scale=1.0) + WhiteKernel(noise_level=0.01, noise_level_bounds=white_kernel_noise_bounds)
gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=100)


data_points_x = exp_data[f"participant_{subject_code}"][f"gpal_results"]["given_numbers"]

data_points_y = exp_data[f"participant_{subject_code}"][f"gpal_results"]["estimates"]

df = pd.DataFrame({"stimulus": data_points_x, "response": data_points_y})


fig, ax = plot_GP(dataframe=df, gp_regressor=gpr)
ax.set_title(f"Subject #{subject_code}\nWhiteKernel noise bound min={repr(white_kernel_noise_bounds[0])}")