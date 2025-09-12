# TODO
# 1. parameter convergence plot
# 2. support rate
# 3. likelihood ratio
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
from utils_simulation import (HypotheticalParticipant, compute_bic,
                              compute_BIC_by_models, fit_model)

sim_config_path = (
    Path(__file__).parent / "simulation_NLE" / "simulation_config.yaml"
)

with sim_config_path.open(encoding="utf-8") as f:
    SIM_CONFIG = yaml.safe_load(f)

N_MAX = SIM_CONFIG["parameter_lists"]["N_MAX"]
N_TRIALS = SIM_CONFIG["parameter_lists"]["N_TRIALS"]

given_number_first = random.randint(1, N_MAX)

x_range = np.linspace(0, N_MAX, N_MAX)

colors = ["red", "orange", "blue", "purple"]
mse_list = []


kernel = RBF(length_scale=2.0) + WhiteKernel(
    noise_level=0.05, noise_level_bounds=(1e-10, 1e1)
)

gp_regressor = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=100)


# GPAL process
true_model_name = "mixed_log_linear"
# true_model_name = "one_cyclic_power"
# true_model_name = "two_cyclic_power"

true_model_likelihood_scipy_minimize = getattr(
    psy_funcs, true_model_name + "_likelihood_scipy_minimize"
)

# Hypothetical experiment for N_TRIALS (GPAL)
X_pred = np.arange(5, N_MAX).reshape(-1, 1)
supported_models_GPAL = []
supported_models_BR = []

for i in tqdm(range(25)):
    print(i)
    participant_GPAL= HypotheticalParticipant(true_model_name)
    print("\nNow running a GPAL experiment simulation")
    optimized_params_list_GPAL = []
    first_given_number = 5 * random.randint(1,100)
    given_number = None

    for ii in range(N_TRIALS):
        print(f"Trial #{ii+1}")
        if ii == 0:
            participant_GPAL.respond(first_given_number)
        else:
            participant_GPAL.respond(given_number)

        X = np.array(participant_GPAL.stimulus_record).reshape(-1, 1)
        y = np.array(participant_GPAL.response_record)

        gp_regressor.fit(X, y)
        log_marginal_likelihood = gp_regressor.log_marginal_likelihood()

        y_pred, y_std = gp_regressor.predict(X_pred, return_std=True)
        #data gen and get a response
        ## find all elements with the maximum std
        max_std_loc = np.argwhere(y_std == np.max(y_std))
        ## randomly select one element
        next_design_loc = max_std_loc[np.random.choice(len(max_std_loc))]
        ## next design
        next_design = np.squeeze(X_pred[next_design_loc])

        given_number = next_design


        # GPAL Parameter convergence
        x0 = []

        model_param_names = list(
            SIM_CONFIG["parameter_lists"]["params_" + true_model_name].keys()
        )

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

        optimized_model = minimize(
            fun=true_model_likelihood_scipy_minimize,
            x0=x0,
            bounds=bounds,
            args=(np.array(participant_GPAL.stimulus_record), np.array(participant_GPAL.response_record)),
            method="L-BFGS-B",
        )

        # parameter convergence record
        optimized_param_values = optimized_model["x"]
        
        optimized_params_list_GPAL.append(optimized_param_values)
        

    # Hypothetical experiment for N_TRIALS (BR)
    print("Now running a BR experiment simulation")
    participant_BR = HypotheticalParticipant(true_model_name)
    optimized_params_list_BR = []
    balanced_given_number = np.linspace(5, 500, N_TRIALS)
    random.shuffle(balanced_given_number)
    for ii in range(N_TRIALS):
        print(f"Trial #{ii+1}")
        participant_BR.respond(balanced_given_number[ii])

        # GPAL Parameter convergence
        x0 = []

        model_param_names = list(
            SIM_CONFIG["parameter_lists"]["params_" + true_model_name].keys()
        )

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

        optimized_model = minimize(
            fun=true_model_likelihood_scipy_minimize,
            x0=x0,
            bounds=bounds,
            args=(np.array(participant_BR.stimulus_record), np.array(participant_BR.response_record)),
            method="L-BFGS-B",
        )

        # parameter convergence record
        optimized_param_values = optimized_model["x"]
        
        optimized_params_list_BR.append(optimized_param_values)    

    # data points visualized
    x = participant_GPAL.stimulus_record
    y = participant_GPAL.response_record    

    fig, ax, _ = fit_model(data_points_x=participant_GPAL.stimulus_record, 
                           data_points_y=participant_GPAL.response_record, 
                           fitting_model_name="mixed_log_linear", 
                           manual_title=f"GPAL data points (#{i+1})")
    # draw true model
    true_params = SIM_CONFIG["parameter_lists"]["params_mixed_log_linear"]
    true_params_noiseless = true_params.copy()
    true_params_noiseless.pop("param_noise")
    true_model = getattr(psy_funcs, true_model_name)
    ax.plot(
        np.linspace(0, N_MAX, N_MAX),
        true_model(
            **true_params_noiseless, given_number=np.linspace(0, N_MAX, N_MAX)
        ),
        color="red",
        label="True Function",
    )
    
    fig.savefig(f"simulation_plots/GPAL_data_points_simulation_{i+1}")

    x = participant_BR.stimulus_record
    y = participant_BR.response_record

    fig, ax, _ = fit_model(data_points_x=participant_BR.stimulus_record, 
                           data_points_y=participant_BR.response_record, 
                           fitting_model_name="mixed_log_linear", 
                           manual_title=f"BR data points (#{i+1})")

    ax.plot(
        np.linspace(0, N_MAX, N_MAX),
        true_model(
            **true_params_noiseless, given_number=np.linspace(0, N_MAX, N_MAX)
        ),
        color="red",
        label="True Function",
    )
    fig.savefig(f"simulation_plots/BR_data_points_simulation_{i+1}")


    # Parameter convergence
    true_params = SIM_CONFIG["parameter_lists"]["params_" + true_model_name]
    print(true_params)
    true_params_values = list(true_params.values())

    param_convergence_GPAL = []
    param_convergence_BR = []
    for ii in range(N_TRIALS):
        optimized_params_GPAL_trial = optimized_params_list_GPAL[ii]
        print(optimized_params_GPAL_trial)

        optimized_params_BR_trial = optimized_params_list_BR[ii]
        print(optimized_params_BR_trial)

        pct_diff_GPAL = []
        for opt_value, true_value in zip(optimized_params_GPAL_trial, true_params_values):
            pct = opt_value / true_value * 100 - 100
            pct = np.clip(pct, -100, 100)
            pct_diff_GPAL.append(pct)
        param_convergence_GPAL.append(pct_diff_GPAL)

        pct_diff_BR = []
        for opt_value, true_value in zip(optimized_params_BR_trial, true_params_values):
            pct = opt_value / true_value * 100 - 100
            pct = np.clip(pct, -100, 100)
            pct_diff_BR.append(pct)
        param_convergence_BR.append(pct_diff_BR)



    # 각 파라미터별로 GPAL vs BR 비교
    fig, ax = plt.subplots(figsize=(6, 4))
    N_PARAMS = len(true_params_values)
    for p in range(N_PARAMS):
        ax.plot([trial[p] for trial in param_convergence_GPAL], label=f'{model_param_names[p]}', linestyle='-')

    ax.axhline(0, color='black', linewidth=1, linestyle=':')  # True value 기준선
    ax.set_xlabel('Trial')
    ax.set_ylabel('% Difference from True Value')
    ax.set_title('Parameter Convergence: GPAL')
    ax.legend()
    ax.grid(True)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    fig.savefig(f"simulation_plots/GPAL_parameter_convergence_simulation_{i+1}")


    print(f"Current Trial {i+1}")
    fig, ax = plt.subplots(figsize=(6, 4))
    N_PARAMS = len(true_params_values)
    for p in range(N_PARAMS):
        ax.plot([trial[p] for trial in param_convergence_BR], label=f'{model_param_names[p]}', linestyle='--')

    ax.axhline(0, color='black', linewidth=1, linestyle=':')  # True value 기준선
    ax.set_xlabel('Trial')
    ax.set_ylabel('% Difference from True Value')
    ax.set_title('Parameter Convergence: BR')
    ax.legend()
    ax.grid(True)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    fig.savefig(f"simulation_plots/BR_parameter_convergence_simulation_{i+1}")


    # BIC support rate (try each of the three models)
    BIC_values_GPAL_by_models = compute_BIC_by_models(participant_GPAL.stimulus_record, participant_GPAL.response_record)
    BIC_values_BR_by_models = compute_BIC_by_models(participant_BR.stimulus_record, participant_BR.response_record)

    supported_model_GPAL = min(BIC_values_GPAL_by_models, key=BIC_values_GPAL_by_models.get)
    supported_model_BR = min(BIC_values_BR_by_models, key=BIC_values_BR_by_models.get)
    supported_models_GPAL.append(supported_model_GPAL)
    supported_models_BR.append(supported_model_BR)
    print(f"Top supported model by GPAL: {supported_model_GPAL}")
    print(supported_models_GPAL)

    count_MLLM_GPAL = supported_models_GPAL.count("MLLM")
    count_1CPM_GPAL = supported_models_GPAL.count("1CPM")
    count_2CPM_GPAL = supported_models_GPAL.count("2CPM")
    print(f"GPAL Support Rate: {count_MLLM_GPAL}/{len(supported_models_GPAL)}")

    print(f"Top supported model by BR: {supported_model_BR}")
    count_MLLM_BR = supported_models_BR.count("MLLM")
    count_1CPM_BR = supported_models_BR.count("1CPM")
    count_2CPM_BR = supported_models_BR.count("2CPM")
    print(f"BR Support Rate: {count_MLLM_BR}/{len(supported_models_BR)}")


    print(supported_models_BR)

# BR Likelihood ratio