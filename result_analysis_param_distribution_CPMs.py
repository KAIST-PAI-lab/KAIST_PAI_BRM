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
import pickle
import time

import simulation_NLE.psychometric_functions as psy_funcs
from utils_simulation import *

exp_data = get_results_data_points(data_dir=Path("data"))

#%%

optimized_params_oneCPM_all = []
optimized_params_oneCPM_ado = []
optimized_params_oneCPM_gpal = []
optimized_params_oneCPM_br = []

for subject_code, results in exp_data.items():

    ado_stimuli = results["ado_results"]["given_numbers"]
    ado_responses = results["ado_results"]["estimates"]

    gpal_stimuli = results["gpal_results"]["given_numbers"]
    gpal_responses = results["gpal_results"]["estimates"]

    br_stimuli = results["random_results"]["given_numbers"]
    br_responses = results["random_results"]["estimates"]

    all_stimuli = ado_stimuli + gpal_stimuli + br_stimuli
    all_responses = ado_responses + gpal_responses + br_responses

    _, _, fitted_oneCPM_all = fit_model(all_stimuli, all_responses, "one_cyclic_power")
    _, _, fitted_oneCPM_ado = fit_model(ado_stimuli, ado_responses, "one_cyclic_power")
    _, _, fitted_oneCPM_gpal = fit_model(gpal_stimuli, gpal_responses, "one_cyclic_power")
    _, _, fitted_oneCPM_br = fit_model(br_stimuli, br_responses, "one_cyclic_power")


    optimized_params_oneCPM_all.append(fitted_oneCPM_all["x"])
    optimized_params_oneCPM_ado.append(fitted_oneCPM_ado["x"])
    optimized_params_oneCPM_gpal.append(fitted_oneCPM_gpal["x"])
    optimized_params_oneCPM_br.append(fitted_oneCPM_br["x"])

#%%

# one cpm histogram

figure, axes = plt.subplots(4, 2, figsize=(8, 15))
n_bins = 10

# ALL
axes[0, 0].hist([_[0] for _ in optimized_params_oneCPM_all], bins=n_bins, edgecolor='black')
mean = round(np.mean([_[0] for _ in optimized_params_oneCPM_all]), 2)
std = round(np.std([_[0] for _ in optimized_params_oneCPM_all], ddof=1), 2)
axes[0, 0].set_title(f"Fitted to all\nOptimized exponent value\nmean={mean}, std={std}")

axes[0, 1].hist([_[1] for _ in optimized_params_oneCPM_all], bins=n_bins, edgecolor='black')
mean = round(np.mean([_[1] for _ in optimized_params_oneCPM_all]), 2)
std = round(np.std([_[1] for _ in optimized_params_oneCPM_all], ddof=1), 2)
axes[0, 1].set_title(f"Fitted to all\nOptimized noise value\nmean={mean}, std={std}")

# ADO
axes[1, 0].hist([_[0] for _ in optimized_params_oneCPM_ado], bins=n_bins, edgecolor='black')
mean = round(np.mean([_[0] for _ in optimized_params_oneCPM_ado]), 2)
std = round(np.std([_[0] for _ in optimized_params_oneCPM_ado], ddof=1), 2)
axes[1, 0].set_title(f"Fitted to ADO block\nOptimized exponent value\nmean={mean}, std={std}")

axes[1, 1].hist([_[1] for _ in optimized_params_oneCPM_ado], bins=n_bins, edgecolor='black')
mean = round(np.mean([_[1] for _ in optimized_params_oneCPM_ado]), 2)
std = round(np.std([_[1] for _ in optimized_params_oneCPM_ado], ddof=1), 2)
axes[1, 1].set_title(f"Fitted to ADO block\nOptimized noise value\nmean={mean}, std={std}")

# GPAL
axes[2, 0].hist([_[0] for _ in optimized_params_oneCPM_gpal], bins=n_bins, edgecolor='black')
mean = round(np.mean([_[0] for _ in optimized_params_oneCPM_gpal]), 2)
std = round(np.std([_[0] for _ in optimized_params_oneCPM_gpal], ddof=1), 2)
axes[2, 0].set_title(f"Fitted to GPAL block\nOptimized exponent value\nmean={mean}, std={std}")

axes[2, 1].hist([_[1] for _ in optimized_params_oneCPM_gpal], bins=n_bins, edgecolor='black')
mean = round(np.mean([_[1] for _ in optimized_params_oneCPM_gpal]), 2)
std = round(np.std([_[1] for _ in optimized_params_oneCPM_gpal], ddof=1), 2)
axes[2, 1].set_title(f"Fitted to GPAL block\nOptimized noise value\nmean={mean}, std={std}")


# BR
axes[3, 0].hist([_[0] for _ in optimized_params_oneCPM_br], bins=n_bins, edgecolor='black')
mean = round(np.mean([_[0] for _ in optimized_params_oneCPM_br]), 2)
std = round(np.std([_[0] for _ in optimized_params_oneCPM_br], ddof=1), 2)
axes[3, 0].set_title(f"Fitted to BR block\nOptimized exponent value\nmean={mean}, std={std}")

axes[3, 1].hist([_[1] for _ in optimized_params_oneCPM_br], bins=n_bins, edgecolor='black')
mean = round(np.mean([_[1] for _ in optimized_params_oneCPM_br]), 2)
std = round(np.std([_[1] for _ in optimized_params_oneCPM_br], ddof=1), 2)
axes[3, 1].set_title(f"Fitted to BR block\nOptimized noise value\nmean={mean}, std={std}")

figure.tight_layout()


#%%

optimized_params_twoCPM_all = []
optimized_params_twoCPM_ado = []
optimized_params_twoCPM_gpal = []
optimized_params_twoCPM_br = []

for subject_code, results in exp_data.items():

    

    ado_stimuli = results["ado_results"]["given_numbers"]
    ado_responses = results["ado_results"]["estimates"]

    gpal_stimuli = results["gpal_results"]["given_numbers"]
    gpal_responses = results["gpal_results"]["estimates"]

    br_stimuli = results["random_results"]["given_numbers"]
    br_responses = results["random_results"]["estimates"]

    all_stimuli = ado_stimuli + gpal_stimuli + br_stimuli
    all_responses = ado_responses + gpal_responses + br_responses

    _, _, fitted_twoCPM_all = fit_model(all_stimuli, all_responses, "two_cyclic_power")
    _, _, fitted_twoCPM_ado = fit_model(ado_stimuli, ado_responses, "two_cyclic_power")
    _, _, fitted_twoCPM_gpal = fit_model(gpal_stimuli, gpal_responses, "two_cyclic_power")
    _, _, fitted_twoCPM_br = fit_model(br_stimuli, br_responses, "two_cyclic_power")


    optimized_params_twoCPM_all.append(fitted_twoCPM_all["x"])
    optimized_params_twoCPM_ado.append(fitted_twoCPM_ado["x"])
    optimized_params_twoCPM_gpal.append(fitted_twoCPM_gpal["x"])
    optimized_params_twoCPM_br.append(fitted_twoCPM_br["x"])

    
# two cpm histogram

figure, axes = plt.subplots(4, 2, figsize=(8, 15))
n_bins = 10

# ALL
axes[0, 0].hist([_[0] for _ in optimized_params_twoCPM_all], bins=n_bins, edgecolor='black')
mean = round(np.mean([_[0] for _ in optimized_params_twoCPM_all]), 2)
std = round(np.std([_[0] for _ in optimized_params_twoCPM_all], ddof=1), 2)
axes[0, 0].set_title(f"Fitted to all\nOptimized exponent value\nmean={mean}, std={std}")

axes[0, 1].hist([_[1] for _ in optimized_params_twoCPM_all], bins=n_bins, edgecolor='black')
mean = round(np.mean([_[1] for _ in optimized_params_twoCPM_all]), 2)
std = round(np.std([_[1] for _ in optimized_params_twoCPM_all], ddof=1), 2)
axes[0, 1].set_title(f"Fitted to all\nOptimized noise value\nmean={mean}, std={std}")

# ADO
axes[1, 0].hist([_[0] for _ in optimized_params_twoCPM_ado], bins=n_bins, edgecolor='black')
mean = round(np.mean([_[0] for _ in optimized_params_twoCPM_ado]), 2)
std = round(np.std([_[0] for _ in optimized_params_twoCPM_ado], ddof=1), 2)
axes[1, 0].set_title(f"Fitted to ADO block\nOptimized exponent value\nmean={mean}, std={std}")

axes[1, 1].hist([_[1] for _ in optimized_params_twoCPM_ado], bins=n_bins, edgecolor='black')
mean = round(np.mean([_[1] for _ in optimized_params_twoCPM_ado]), 2)
std = round(np.std([_[1] for _ in optimized_params_twoCPM_ado], ddof=1), 2)
axes[1, 1].set_title(f"Fitted to ADO block\nOptimized noise value\nmean={mean}, std={std}")

# GPAL
axes[2, 0].hist([_[0] for _ in optimized_params_twoCPM_gpal], bins=n_bins, edgecolor='black')
mean = round(np.mean([_[0] for _ in optimized_params_twoCPM_gpal]), 2)
std = round(np.std([_[0] for _ in optimized_params_twoCPM_gpal], ddof=1), 2)
axes[2, 0].set_title(f"Fitted to GPAL block\nOptimized exponent value\nmean={mean}, std={std}")

axes[2, 1].hist([_[1] for _ in optimized_params_twoCPM_gpal], bins=n_bins, edgecolor='black')
mean = round(np.mean([_[1] for _ in optimized_params_twoCPM_gpal]), 2)
std = round(np.std([_[1] for _ in optimized_params_twoCPM_gpal], ddof=1), 2)
axes[2, 1].set_title(f"Fitted to GPAL block\nOptimized noise value\nmean={mean}, std={std}")


# BR
axes[3, 0].hist([_[0] for _ in optimized_params_twoCPM_br], bins=n_bins, edgecolor='black')
mean = round(np.mean([_[0] for _ in optimized_params_twoCPM_br]), 2)
std = round(np.std([_[0] for _ in optimized_params_twoCPM_br], ddof=1), 2)
axes[3, 0].set_title(f"Fitted to BR block\nOptimized exponent value\nmean={mean}, std={std}")

axes[3, 1].hist([_[1] for _ in optimized_params_twoCPM_br], bins=n_bins, edgecolor='black')
mean = round(np.mean([_[1] for _ in optimized_params_twoCPM_br]), 2)
std = round(np.std([_[1] for _ in optimized_params_twoCPM_br], ddof=1), 2)
axes[3, 1].set_title(f"Fitted to BR block\nOptimized noise value\nmean={mean}, std={std}")

figure.tight_layout()

# %%
