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

import simulation_NLE.psychometric_functions as psy_funcs
from utils_simulation import (fit_model, get_GP_mean_function,
                              get_results_data_points, run_residual_analysis,
                              run_residual_analysis_GP)

results = get_results_data_points()

subject_codes = [subject_code for subject_code in results.keys()]

design_names = ["ado_results", "gpal_results", "random_results"]

subject = subject_codes[4]
x_points = results[subject]["random_results"]["given_numbers"]
y_points = results[subject]["random_results"]["estimates"]


#%% 
model_name = "mixed_log_linear"

optimized_parameters_list = []

optimized_parameters_list_each_design = [[] for _ in range(3)] 

# each subject
for subject_code in subject_codes:
    x_data_points_subject = []
    y_data_points_subject = []
    for design_name_index in range(len(design_names)):
        design_name = design_names[design_name_index]

        x_data_points_each_design = results[subject_code][design_name]["given_numbers"]
        x_data_points_subject.extend(x_data_points_each_design)
        y_data_points_each_design = results[subject_code][design_name]["estimates"]
        y_data_points_subject.extend(y_data_points_each_design)
        
        fig, ax, optimized_model = fit_model(data_points_x=x_data_points_each_design,
                                            data_points_y=y_data_points_each_design,
                                            fitting_model_name=model_name,
                                            manual_title=subject_code + f" ({design_name})")
        
        optimized_parameters=optimized_model["x"][:-1]
        optimized_parameters_list_each_design[design_name_index].append(optimized_model["x"]) 

        fig_resi, ax_resi, residual_list = run_residual_analysis(
            data_points_x=x_data_points_each_design,
            data_points_y=y_data_points_each_design,
            model_name=model_name,
            parameters=optimized_parameters,
            manual_title=subject_code + f" ({design_name})")
        
        fig.savefig(f"plots/{subject_code}_{design_name}_{model_name}.png", bbox_inches="tight")
        fig_resi.savefig(f"plots/{subject_code}_{design_name}_residual_analysis.png",  bbox_inches="tight")

    print(len(x_data_points_subject))

    fig, ax, optimized_model = fit_model(data_points_x=x_data_points_subject,
                                            data_points_y=y_data_points_subject,
                                            fitting_model_name=model_name,
                                            manual_title=subject_code)
        
    optimized_parameters= optimized_model["x"][:-1]
    optimized_parameters_list.append(optimized_model["x"])

    fig_resi, ax_resi, residual_list = run_residual_analysis(
        data_points_x=x_data_points_subject,
        data_points_y=y_data_points_subject,
        model_name=model_name,
        parameters=optimized_parameters,
        manual_title=subject_code)


    fig.savefig(f"plots/{subject_code}_{model_name}.png", bbox_inches="tight")
    fig_resi.savefig(f"plots/{subject_code}_residual_analysis.png", bbox_inches="tight")

#%%
# parameter distribution each participant
slopes = [params[0] for params in optimized_parameters_list]
intercepts = [params[1] for params in optimized_parameters_list]
log_mixes = [params[2] for params in optimized_parameters_list]
noises = [params[3] for params in optimized_parameters_list]


params_list = [slopes, intercepts, log_mixes, noises]
param_names = ["Slope", "Intercept", "Log-Mix", "Noise"]

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for ax, values, name in zip(axes, params_list, param_names):
    mean_val = np.mean(values)
    std_val = np.std(values, ddof=1)
    ax.hist(values, bins=30, color="skyblue", edgecolor="black", alpha=0.7)
    ax.set_title(f"{name}\nmean={mean_val:.2f}, std={std_val:.2f}")
    ax.set_ylabel("Frequency", fontsize=10)
    ax.yaxis.set_major_locator(MultipleLocator(1))

plt.suptitle("Optimized Parameters Distribution", fontsize=16, y=1.05) 
plt.tight_layout()
plt.show()
#%%
# parameter distribution each design

optimized_parameters_list_ADO = optimized_parameters_list_each_design[0]
optimized_parameters_list_GPAL = optimized_parameters_list_each_design[1]
optimized_parameters_list_BR  = optimized_parameters_list_each_design[2]

designs = {
    "ADO": optimized_parameters_list_ADO,
    "GPAL": optimized_parameters_list_GPAL,
    "BR": optimized_parameters_list_BR
}

param_names = ["Slope", "Intercept", "Log Mix", "Noise"]

for design_name, param_list in designs.items():
    slopes     = [p[0] for p in param_list]
    intercepts = [p[1] for p in param_list]
    log_mixes  = [p[2] for p in param_list]
    noises     = [p[3] for p in param_list]

    params_list = [slopes, intercepts, log_mixes, noises]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, values, name in zip(axes, params_list, param_names):
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1)
        
        ax.hist(values, bins=30, color="skyblue", edgecolor="black", alpha=0.7)
        ax.set_title(f"{name}\nmean={mean_val:.2f}, std={std_val:.2f}", fontsize=11)
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.tick_params(axis="both", labelsize=9)
        ax.set_ylabel("Frequency", fontsize=10)

    plt.tight_layout()
    plt.suptitle(f"Optimized Parameters Distribution ({design_name})", fontsize=14, y=1.05)
    plt.show()



#%%
# Total
data_points_x_exhaustive = []
data_points_y_exhaustive = []
for subject_code in subject_codes:
    for design_name in design_names:
        x_points = results[subject_code][design_name]["given_numbers"]
        y_points = results[subject_code][design_name]["estimates"]
        data_points_x_exhaustive.extend(x_points)
        data_points_y_exhaustive.extend(y_points)

fig, ax, optimized_model = fit_model(data_points_x_exhaustive, data_points_y_exhaustive, "mixed_log_linear", manual_title=f"All Data Points (len={len(data_points_x_exhaustive)})")

fig.savefig(f"plots/exhaustive_{model_name}.png")

optimized_parameters=optimized_model["x"][:-1]

fig_resi, ax_resi, residual_list = run_residual_analysis(
    data_points_x = data_points_x_exhaustive,
    data_points_y = data_points_y_exhaustive,
    model_name="mixed_log_linear", 
    parameters=optimized_parameters,
    manual_title=f"All Data Points (len={len(data_points_x_exhaustive)})")


fig.savefig(f"plots/exhaustive_{model_name}.png", bbox_inches="tight")
fig_resi.savefig(f"plots/exhaustive_residual_analysis.png", bbox_inches="tight")


# %%
