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
from utils_simulation import (HypotheticalParticipant, compute_bic,
                              compute_BIC_by_models, fit_model, save_pickle_checkpoint)

N_TRIALS = 50

checkpoint_num = 6

exponent_value_cases = np.arange(0, 1.61, 0.01) # interval 0.01

params_cases_total = []
for exponent_value in exponent_value_cases:
            params_case = [exponent_value]
            params_cases_total.append(params_case)

print(len(params_cases_total))

# ---
kernel = C(1.0)*RBF(length_scale=1.0) + WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-05, 1e5))
gp_regressor = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=100)

# ---
true_model_name = "two_cyclic_power"


sim_config_path = (
    Path(__file__).parent / "simulation_NLE" / "simulation_config.yaml"
)

with sim_config_path.open(encoding="utf-8") as f:
    SIM_CONFIG = yaml.safe_load(f)

model_param_names = list(
            SIM_CONFIG["parameter_lists"]["params_" + true_model_name].keys()
        )
N_MAX = SIM_CONFIG["parameter_lists"]["N_MAX"]


true_model_likelihood_scipy_minimize = getattr(
    psy_funcs, true_model_name + "_likelihood_scipy_minimize"
)

# Model optimization parameters
x0 = []

for name in model_param_names:
    start_value = SIM_CONFIG["parameter_lists"][
        "params_optimize_" + true_model_name
    ][name + "_range_start"]

    end_value = SIM_CONFIG["parameter_lists"]["params_optimize_" + true_model_name][
        name + "_range_end"
    ]

    choices = np.round(np.arange(start_value, end_value, 0.1), 1)

    # Randomly select one value
    random_value = np.random.choice(choices)

    x0.append(random_value)

bounds = []
for name in model_param_names:
    bounds.append(
        (
            SIM_CONFIG["parameter_lists"]["params_optimize_" + true_model_name][
                name + "_range_start"
            ],
            SIM_CONFIG["parameter_lists"]["params_optimize_" + true_model_name][
                name + "_range_end"
            ],
        )
    )

total_iter_number = len(params_cases_total)

true_param_combinations = []

GPAL_param_combinations = []
GPAL_param_combinations_across_trials = [[] for _ in range(N_TRIALS)]
BR_param_combinations = []
BR_param_combinations_across_trials = [[] for _ in range(N_TRIALS)]

force_skip = 120
error_occured = {}
for case_idx, exponent_value in tqdm(
    enumerate(params_cases_total), total=len(params_cases_total)
):
    if case_idx <= force_skip:
        continue
    
    try:
        time_point_0 = time.time()

        current_parameter_combination = [exponent_value]

        participant_GPAL = HypotheticalParticipant(true_model_name)
        
        first_given_number = 5*random.randint(1, 101)

        given_number = None

        # generate data (GPAL)
        params_override = None

        # plot_model(model_name=true_model_name, parameters=current_parameter_combination)

        for i in range(N_TRIALS):
            # print(f"GPAL Trial #{i+1}")

            if i == 0:
                stimulus = first_given_number
            else:
                stimulus = given_number

            noise_value = stimulus * 0.1

            # print(f"Stimulus: {stimulus}, Noise: {noise_value}")

            trial_param_combination = [exponent_value, noise_value]
            true_param_combinations.append(trial_param_combination)
            
            params_override = {param_name: param_value for param_name, param_value in zip(model_param_names, trial_param_combination)}
            
            # print(f"Current True Parameters: {params_override}")

            participant_GPAL.respond(stimulus, params_override=params_override)

            X = np.array(participant_GPAL.stimulus_record).reshape(-1, 1)
            y = np.array(participant_GPAL.response_record)

            gp_regressor.fit(X, y)

            X_pred = np.arange(5, N_MAX).reshape(-1, 1)
            y_pred, y_std = gp_regressor.predict(X_pred, return_std=True)
            #data gen and get a response
            ## find all elements with the maximum std
            max_std_loc = np.argwhere(y_std == np.max(y_std))
            ## randomly select one element
            next_design_loc = max_std_loc[np.random.choice(len(max_std_loc))]
            ## next design
            next_design = np.squeeze(X_pred[next_design_loc])

            given_number = next_design

            optimized_model_GPAL_each_trial = minimize(
            fun=true_model_likelihood_scipy_minimize,
            x0=x0,
            bounds=bounds,
            args=(np.array(participant_GPAL.stimulus_record), np.array(participant_GPAL.response_record)),
            method="L-BFGS-B",
            )

            GPAL_param_combinations_across_trials[i].append(optimized_model_GPAL_each_trial["x"])

        # save checkpoint
        optimized_params_across_trials_per_param_comb_GPAL = [params_per_trial[-1] for params_per_trial in GPAL_param_combinations_across_trials] # optimized parameters across trials for this parameter combination
        # plot_data_points(participant_GPAL.stimulus_record, participant_GPAL.response_record)

        checkpoint_GPAL = {"design_optimization": "GPAL", 
                        "parameters": current_parameter_combination, 
                        "stimuli": participant_GPAL.stimulus_record, 
                        "responses": participant_GPAL.response_record, 
                        "optimized_parameters": optimized_params_across_trials_per_param_comb_GPAL}
        checkpoint_path_GPAL = f"simulation_checkpoints/GPAL_param_robustness_#{checkpoint_num}/records_GPAL"
        
        checkpoint_file_name_GPAL = f"checkpoint_GPAL_param_#{case_idx}_({'-'.join(map(str, current_parameter_combination))}).pkl"
        
        save_pickle_checkpoint(checkpoint_path_GPAL, checkpoint_file_name_GPAL, checkpoint_GPAL)


        # generate data (BR)
        participant_BR = HypotheticalParticipant(true_model_name)

        balanced_given_number = np.linspace(5, 500, N_TRIALS)

        random.shuffle(balanced_given_number)

        params_override = None
        
        for i in range(N_TRIALS):
            # print(f"BR Trial #{i+1}")
            stimulus = balanced_given_number[i]

            noise_value = stimulus * 0.1
            # print(f"Noise value before clipping: {noise_value}")

            current_param_combination = [exponent_value, noise_value]

            params_override = {param_name: param_value for param_name, param_value in zip(model_param_names, current_param_combination)}

            participant_BR.respond(stimulus=stimulus, params_override=params_override)

            optimized_model_BR_each_trial = minimize(
            fun=true_model_likelihood_scipy_minimize,
            x0=x0,
            bounds=bounds,
            args=(np.array(participant_BR.stimulus_record), np.array(participant_BR.response_record)),
            method="L-BFGS-B",
            )

            BR_param_combinations_across_trials[i].append(optimized_model_BR_each_trial["x"])

        # save checkpoint BR
        optimized_params_across_trials_per_param_comb_BR = [params_per_trial[-1] for params_per_trial in BR_param_combinations_across_trials] # optimized parameters across trials for this parameter combination
        # plot_data_points(participant_GPAL.stimulus_record, participant_GPAL.response_record)

        checkpoint_BR = {"design_optimization": "BR", 
                        "parameters": current_parameter_combination, 
                        "stimuli": participant_BR.stimulus_record, 
                        "responses": participant_BR.response_record, 
                        "optimized_parameters": optimized_params_across_trials_per_param_comb_BR}
        checkpoint_path_BR = f"simulation_checkpoints/GPAL_param_robustness_#{checkpoint_num}/records_BR"
        
        checkpoint_file_name_BR = f"checkpoint_BR_param_#{case_idx}_({'-'.join(map(str, current_parameter_combination))}).pkl"
        
        save_pickle_checkpoint(checkpoint_path_BR, checkpoint_file_name_BR, checkpoint_BR)

        checkpoint_check_path = full_path = os.path.join(checkpoint_path_BR, checkpoint_file_name_BR)

        # GPAL Parameter convergence
        optimized_model_GPAL = minimize(
            fun=true_model_likelihood_scipy_minimize,
            x0=x0,
            bounds=bounds,
            args=(np.array(participant_GPAL.stimulus_record), np.array(participant_GPAL.response_record)),
            method="L-BFGS-B",
        )

        optimized_model_BR = minimize(
            fun=true_model_likelihood_scipy_minimize,
            x0=x0,
            bounds=bounds,
            args=(np.array(participant_BR.stimulus_record), np.array(participant_BR.response_record)),
            method="L-BFGS-B",
        )

        GPAL_param_combination = optimized_model_GPAL["x"]
        GPAL_param_combinations.append(GPAL_param_combination)
        
        BR_param_combination = optimized_model_BR["x"]
        BR_param_combinations.append(BR_param_combination)

        # print(f"True parameter combinations currently done: {true_param_combinations}")
        # print(f"GPAL current collected paramter combinations: {GPAL_param_combinations}")
        # print(f"BR current collected parameter combinations: {BR_param_combinations}")

        time_point_1 = time.time()
        print(f"Loop {case_idx} took: {time_point_1 - time_point_0:.3f} seconds")
    except Exception as e:
        error_occured[f"{case_idx}"] = e
        continue

print(error_occured)
         

# #%%
# all_log_mix_values = np.tile(log_mix_value_cases, 3)

# pearson_values_across_trials_GPAL = []

# array1 = None
# array2 = None

# fig, ax = plt.subplots(figsize = (6,4))

# for i in range(N_TRIALS):
#     array1 = all_log_mix_values
#     print(array1)
#     array2 = [_[2] for _ in GPAL_param_combinations_across_trials[i]]
#     print(array2)
#     pearson_value = np.corrcoef(array1, array2)[0, 1]
#     pearson_values_across_trials_GPAL.append(pearson_value)

# ax.scatter(array1, array2)

# ax.set_xlabel("True Lambda Values")
# ax.set_ylabel("Estimated Lambda Values")

# ax.set_title("true lambda and estimated lambda values (GPAL, Trial #50)")

# fig.savefig(f"plots/true lambda and estimated lambda values (GPAL)")

# #%%
# pearson_values_across_trials_BR = []

# fig, ax = plt.subplots(figsize = (6,4))

# for i in range(N_TRIALS):
#     array1 = all_log_mix_values
#     array2 = [_[2] for _ in BR_param_combinations_across_trials[i]]
#     pearson_value = np.corrcoef(array1, array2)[0, 1]
#     pearson_values_across_trials_BR.append(pearson_value)

# ax.set_xlabel("True Lambda Values")
# ax.set_ylabel("Estimated Lambda Values")
# ax.scatter(array1, array2)

# ax.set_title("true lambda and estimated lambda values (BR, Trial #50)")
# fig.savefig(f"plots/true lambda and estimated lambda values (BR, Trial #50)")

# print(pearson_values_across_trials_GPAL)
# print(pearson_values_across_trials_BR)
# #%%
# trials = range(1, N_TRIALS+1)

# plt.plot(trials, pearson_values_across_trials_GPAL, label="GPAL", linestyle="-")
# plt.plot(trials, pearson_values_across_trials_BR, label="BR", linestyle="--")

# plt.axhline(0, color="black", linewidth=1, linestyle=":")  # 기준선
# plt.ylim(0, 1)  # Pearson correlation 범위
# plt.xlabel("Trial")
# plt.ylabel("Pearson correlation (exponent)")
# plt.title("Correlation of exponent parameter across trials (1CPM)")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# ax = plt.gca()
# ax.xaxis.set_major_locator(MultipleLocator(1))
# plt.show()

# # param robustness pearson values
# # true_param_combinations 
# # GPAL_param_combinations 
# # BR_param_combinations 

# true_param_exponent_values = [_[0] for _ in true_param_combinations]

# GPAL_param_exponent_values = [_[0] for _ in GPAL_param_combinations]

# BR_param_exponent_values = [_[0] for _ in BR_param_combinations]

# true_to_GPAL_pearson_values = []
# true_to_BR_pearson_values = []

# true_to_GPAL_exponent_pearson_value = np.corrcoef(true_param_exponent_values, GPAL_param_exponent_values)[0,1]

# true_to_BR_exponent_pearson_value = np.corrcoef(true_param_exponent_values, BR_param_exponent_values)[0,1] 

# true_to_GPAL_pearson_values.append(true_to_GPAL_exponent_pearson_value)

# true_to_BR_pearson_values.append(true_to_BR_exponent_pearson_value)

# print("True to GPAL:", true_to_GPAL_pearson_values)
# print("True to BR:", true_to_BR_pearson_values)


# # %%
