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

exp_data = get_results_data_points()

subject_code = "25081110"


# load data points from the target subject

figure, axes = plt.subplots (1, 3, figsize = (12, 4))

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

# ADO
ADO_data_points_x = exp_data[f"participant_{subject_code}"][f"ado_results"]["given_numbers"]
ADO_data_points_x_jittered = ADO_data_points_x + np.random.normal(0, 10, size=len(ADO_data_points_x))

ADO_data_points_y = exp_data[f"participant_{subject_code}"][f"ado_results"]["estimates"]
axes[0].scatter(ADO_data_points_x_jittered, ADO_data_points_y, s=data_points_size, alpha=data_points_opacity, color="black")
axes[0].set_xlim(-50, 550)
axes[0].set_ylim(-50, 550)
axes[0].set_title("ADO", fontsize=font_size_title)

# ADO - fit only MLLM
_, _, optimized_MLLM_ADO = fit_model(ADO_data_points_x, ADO_data_points_y, "mixed_log_linear")
x_range = np.arange(0, int(N_MAX) + 1)
y_pred = MLLM(*optimized_MLLM_ADO["x"][:-1], x_range)
axes[0].plot(x_range, y_pred, linewidth=2, label = "MLLM (fitted)")

axes[0].set_xlabel("Given Number", fontsize=font_size_axis_labels)
axes[0].set_ylabel("Estimate", fontsize=font_size_axis_labels)
axes[0].tick_params(axis="both", which="major", labelsize=font_size_tick_size)

axes[0].legend(loc=legend_location, fontsize= font_size_legend)


# GPAL
GPAL_data_points_x = exp_data[f"participant_{subject_code}"][f"gpal_results"]["given_numbers"]
GPAL_data_points_y = exp_data[f"participant_{subject_code}"][f"gpal_results"]["estimates"]
axes[1].scatter(GPAL_data_points_x, GPAL_data_points_y, s=data_points_size, alpha=data_points_opacity, color="black")
axes[1].set_xlim(-50, 550)
axes[1].set_ylim(-50, 550)
axes[1].set_title("GPAL", fontsize=font_size_title)

# GPAL - fit MLLM
# _, _, optimized_MLLM_GPAL = fit_model(GPAL_data_points_x, GPAL_data_points_y, "mixed_log_linear")
# x_range = np.arange(0, int(N_MAX) + 1)
# y_pred = MLLM(*optimized_MLLM_GPAL["x"][:-1], x_range)
# axes[1].plot(x_range, y_pred, linewidth=2, label = "MLLM (fitted)")

# GPAL - fit 1CPM
# _, _, optimized_OneCPM_GPAL = fit_model(GPAL_data_points_x, GPAL_data_points_y, "one_cyclic_power")
# x_range = np.arange(0, int(N_MAX) + 1)
# y_pred = OneCPM(*optimized_OneCPM_GPAL["x"][:-1], x_range)
# axes[1].plot(x_range, y_pred, linewidth=2, label = "1CPM (fitted)")

# GPAL - GP mean function (+ confidence region)
white_kernel_noise_bounds = (0.4, 1e5)
kernel = C(1.0)*RBF(length_scale=1.0) + WhiteKernel(noise_level=0.01, noise_level_bounds=white_kernel_noise_bounds)
gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=100)
_, _, fitted_gpr_GPAL = get_GP_mean_function(GPAL_data_points_x, GPAL_data_points_y, gpr)

gp_mean_func_GPAL, gp_std_GPAL = fitted_gpr_GPAL.predict(np.array(x_range).reshape(-1, 1), return_std=True)

axes[1].plot(x_range, gp_mean_func_GPAL, label="GP Mean Function")
axes[1].fill_between(
    x_range,
    gp_mean_func_GPAL - gp_std_GPAL,
    gp_mean_func_GPAL + gp_std_GPAL,
    alpha=0.3,
)
axes[1].set_xlabel("Given Number", fontsize=font_size_axis_labels)
axes[1].set_ylabel("Estimate", fontsize=font_size_axis_labels)
axes[1].tick_params(axis="both", which="major", labelsize=font_size_tick_size)

axes[1].legend(loc=legend_location, fontsize= font_size_legend)

# BR
BR_data_points_x = exp_data[f"participant_{subject_code}"][f"random_results"]["given_numbers"]
BR_data_points_y = exp_data[f"participant_{subject_code}"][f"random_results"]["estimates"]
axes[2].scatter(BR_data_points_x, BR_data_points_y, s=data_points_size, alpha=data_points_opacity, color="black")
axes[2].set_xlim(-50, 550)
axes[2].set_ylim(-50, 550)
axes[2].set_title("BR", fontsize=font_size_title)

# BR - fit MLLM 
# _, _, optimized_MLLM_BR = fit_model(BR_data_points_x, BR_data_points_y, "mixed_log_linear")
# x_range = np.arange(0, int(N_MAX) + 1)
# y_pred = MLLM(*optimized_MLLM_BR["x"][:-1], x_range)
# axes[2].plot(x_range, y_pred, linewidth=2, label = "MLLM (fitted)")

# BR - fit 1CPM
# _, _, optimized_OneCPM_BR = fit_model(BR_data_points_x, BR_data_points_y, "one_cyclic_power")
# x_range = np.arange(0, int(N_MAX) + 1)
# y_pred = OneCPM(*optimized_OneCPM_BR["x"][:-1], x_range)
# axes[2].plot(x_range, y_pred, linewidth=2, label = "1CPM (fitted)")

# BR - GP mean function (+ confidence region)
kernel = C(1.0)*RBF(length_scale=1.0) + WhiteKernel(noise_level=0.01, noise_level_bounds=white_kernel_noise_bounds)
gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=100)
_, _, fitted_gpr_BR = get_GP_mean_function(BR_data_points_x, BR_data_points_y, gpr)
gp_mean_func_BR, gp_std_BR = fitted_gpr_BR.predict(np.array(x_range).reshape(-1, 1), return_std=True)

axes[2].set_xlabel("Given Number", fontsize=font_size_axis_labels)
axes[2].set_ylabel("Estimate", fontsize=font_size_axis_labels)
axes[2].tick_params(axis="both", which="major", labelsize=font_size_tick_size)

axes[2].plot(x_range, gp_mean_func_BR, label="GP Mean Function")
axes[2].fill_between(
    x_range,
    gp_mean_func_BR - gp_std_BR,
    gp_mean_func_BR + gp_std_BR,
    alpha=0.3,
)

axes[2].legend(loc=legend_location, fontsize= font_size_legend)

figure.tight_layout()
figure.savefig("plots/plotting_task_#0")
# %%
