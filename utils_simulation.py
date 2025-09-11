from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize

import simulation_NLE.psychometric_functions as psy_funcs

global SIM_CONFIG


sim_config_path = (
            Path(__file__).parent / "simulation_NLE" / "simulation_config.yaml"
        )

with sim_config_path.open(encoding="utf-8") as f:
    SIM_CONFIG = yaml.safe_load(f)

global N_MAX
N_MAX = SIM_CONFIG["parameter_lists"]["N_MAX"]

class HypotheticalParticipant:
    def __init__(self, true_model_name):
        """
        A hypothetical participant class for simulation
        Args:
            true_model_name (_str_): "mixed_log_linear", "one_cyclic_power", "two_cyclic_power"
        """
        self.stimulus_record = []
        self.response_record = []
        self.true_model_name = true_model_name
        self.generate_response = getattr(psy_funcs, true_model_name + "_generate_response")
        
        sim_config_path = (
            Path(__file__).parent / "simulation_NLE" / "simulation_config.yaml"
        )
        
        with sim_config_path.open(encoding="utf-8") as f:
            self.SIM_CONFIG = yaml.safe_load(f)

        self.true_model_params = self.SIM_CONFIG["parameter_lists"]["params_" + self.true_model_name]
        
    def respond(self, stimulus, **override_params):

        params = {**self.true_model_params, **override_params}
        
        response = self.generate_response(**params, given_number=stimulus)

        print(f"Participant responds with {response} to the stimulus {stimulus}")

        self.stimulus_record.append(stimulus)
        self.response_record.append(response)

        return response



def compute_bic(optimized_result, num_data_points):
    """
    optimized_result: scipy.optimize.minimize object
    num_data_points: number of data points (len(simulated_estimate))
    """
    k = len(optimized_result.x)  # param numbers
    nll = optimized_result.fun  # minimized negative log likelihood
    bic = k * np.log(num_data_points) + 2 * nll

    return bic


def compute_BIC_by_models(stimulus_list, response_list):
    
    # load config
    sim_config_path = sim_config_path = (
        Path(__file__).parent / "simulation_NLE" / "simulation_config.yaml"
    )
    with sim_config_path.open(encoding="utf-8") as f:
        SIM_CONFIG = yaml.safe_load(f)
    # Optimize MLLM
    # MLLM params
    MLLM_model_param_names = list(
        SIM_CONFIG["parameter_lists"]["params_mixed_log_linear"].keys()
    )
    # x0 of MLLM
    x0_MLLM = []
    for name in MLLM_model_param_names:
        start_value = SIM_CONFIG["parameter_lists"][
            "params_optimize_mixed_log_linear"
        ][name + "_range_start"]

        end_value = SIM_CONFIG["parameter_lists"][
            "params_optimize_mixed_log_linear"
        ][name + "_range_end"]

        choices = np.round(np.arange(start_value, end_value, 0.1), 1)

        # Randomly select one value
        random_value = np.random.choice(choices)

        x0_MLLM.append(random_value)

    # bounds of MLLM
    bounds_MLLM = []
    for name in MLLM_model_param_names:
        bounds_MLLM.append(
            (
                SIM_CONFIG["parameter_lists"][
                    "params_optimize_mixed_log_linear"
                ][name + "_range_start"],
                SIM_CONFIG["parameter_lists"][
                    "params_optimize_mixed_log_linear"
                ][name + "_range_end"],
            )
        )

    # Optimized MLLM
    optimized_mixed_log_linear = minimize(
        fun=psy_funcs.mixed_log_linear_likelihood_scipy_minimize,
        x0=x0_MLLM,
        bounds=bounds_MLLM,
        args=(np.array(stimulus_list), np.array(response_list)),
        method="L-BFGS-B",
    )

    # Optimize 1CPM
    # 1CPM params
    oneCPM_model_param_names = list(
        SIM_CONFIG["parameter_lists"]["params_one_cyclic_power"].keys()
    )

    # x0 of 1CPM
    x0_oneCPM = []
    for name in oneCPM_model_param_names:
        start_value = SIM_CONFIG["parameter_lists"][
            "params_optimize_one_cyclic_power"
        ][name + "_range_start"]

        end_value = SIM_CONFIG["parameter_lists"][
            "params_optimize_one_cyclic_power"
        ][name + "_range_end"]
        choices = np.round(np.arange(start_value, end_value, 0.1), 1)

        # Randomly select one value
        random_value = np.random.choice(choices)

        x0_oneCPM.append(random_value)

    # bounds of 1CPM
    bounds_oneCPM = []
    for name in oneCPM_model_param_names:
        bounds_oneCPM.append(
            (
                SIM_CONFIG["parameter_lists"][
                    "params_optimize_one_cyclic_power"
                ][name + "_range_start"],
                SIM_CONFIG["parameter_lists"][
                    "params_optimize_one_cyclic_power"
                ][name + "_range_end"],
            )
        )

    # Optimized 1CPM
    optimized_one_cyclic_power = minimize(
        fun=psy_funcs.one_cyclic_power_likelihood_scipy_minimize,
        x0=x0_oneCPM,
        bounds=bounds_oneCPM,
        args=(np.array(stimulus_list), np.array(response_list)),
        method="L-BFGS-B",
    )

    # Optimize 2CPM
    # 2CPM params
    twoCPM_model_param_names = list(
        SIM_CONFIG["parameter_lists"]["params_two_cyclic_power"].keys()
    )

    # x0 of 2CPM
    x0_twoCPM = []
    for name in twoCPM_model_param_names:
        start_value = SIM_CONFIG["parameter_lists"][
            "params_optimize_two_cyclic_power"
        ][name + "_range_start"]

        end_value = SIM_CONFIG["parameter_lists"][
            "params_optimize_two_cyclic_power"
        ][name + "_range_end"]

        choices = np.round(np.arange(start_value, end_value, 0.1), 1)

        # Randomly select one value
        random_value = np.random.choice(choices)

        x0_twoCPM.append(random_value)

    # Bounds of 2CPM
    bounds_twoCPM = []
    for name in twoCPM_model_param_names:
        bounds_twoCPM.append(
            (
                SIM_CONFIG["parameter_lists"][
                    "params_optimize_two_cyclic_power"
                ][name + "_range_start"],
                SIM_CONFIG["parameter_lists"][
                    "params_optimize_two_cyclic_power"
                ][name + "_range_end"],
            )
        )

    # Optimized 2CPM
    optimized_two_cyclic_power = minimize(
        fun=psy_funcs.two_cyclic_power_likelihood_scipy_minimize,
        x0=x0_twoCPM,
        bounds=bounds_twoCPM,
        args=(np.array(stimulus_list), np.array(response_list)),
        method="L-BFGS-B",
    )

    # BIC comparison plots
    bic_mllm = compute_bic(
        optimized_mixed_log_linear, len(stimulus_list)
    )
    bic_1cpm = compute_bic(
        optimized_one_cyclic_power, len(stimulus_list)
    )
    bic_2cpm = compute_bic(
        optimized_two_cyclic_power, len(stimulus_list)
    )

    result = {"MLLM": bic_mllm, "1CPM": bic_1cpm, "2CPM": bic_2cpm}

    return result


def plot_model_fitting(data_points_x, data_points_y, fitting_model_name):
    # model fitting using the scipy minimize
    model_param_names = list(
        SIM_CONFIG["parameter_lists"]["params_" + fitting_model_name].keys()
    )
    
    true_model_likelihood_scipy_minimize = getattr(
        psy_funcs, fitting_model_name + "_likelihood_scipy_minimize"
    )

    x0 = []
    for name in model_param_names:
        start_value = SIM_CONFIG["parameter_lists"][
            "params_optimize_" + fitting_model_name
        ][name + "_range_start"]

        end_value = SIM_CONFIG["parameter_lists"]["params_optimize_" + fitting_model_name][
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
                SIM_CONFIG["parameter_lists"]["params_optimize_" + fitting_model_name][
                    name + "_range_start"
                ],
                SIM_CONFIG["parameter_lists"]["params_optimize_" + fitting_model_name][
                    name + "_range_end"
                ],
            )
        )
    print(f"Optimizing {fitting_model_name} to {len(data_points_x)} X data points and {len(data_points_y)} Y data points")
    print(f"x0: {x0}")
    print(f"bounds: {bounds}")

    optimized_model = minimize(
        fun=true_model_likelihood_scipy_minimize,
        x0=x0,
        bounds=bounds,
        args=(np.array(data_points_x), np.array(data_points_y)),
        method="L-BFGS-B",
    )

    optimized_parameters = optimized_model["x"]
    
    true_model = getattr(psy_funcs, fitting_model_name)
    
    x_range = np.arange(0, N_MAX+1)

    y_pred = true_model(*optimized_parameters[:-1], x_range)
    
    fig, ax = plt.figure(figsize=(6,4))

    plt.plot(x_range, y_pred, label=f"{fitting_model_name}\n{optimized_parameters}")
    plt.scatter(data_points_x, data_points_y)


    return fig, ax, optimized_model

    # plot the datapoints and the fitted model in a figure   


# def plot_GP_mean_function(gp_regressor, data_points_x, data_points_y):




def get_results_data_points(data_dir=Path("data")):
    """
    parsing function for our experiment results
    Return:
        dict: {
            "<participant_folder_name>": {
                "ado_results":   {"given_numbers": [...], "estimates": [...]},
                "gpal_results":  {"given_numbers": [...], "estimates": [...]},
                "random_results":{"given_numbers": [...], "estimates": [...]},
            },
            ...
        }
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    results = defaultdict(dict)
    keys = ("ado_results", "gpal_results", "random_results")

    for folder in data_dir.iterdir():
        if not folder.is_dir():
            continue
        print(f"Current Folder: {folder.name}")

        for file in folder.iterdir():
            if not file.is_file() or file.suffix.lower() != ".csv":
                continue

            file_name = file.name
            print(f"Current File: {file_name}")

            key = next((k for k in keys if k in file_name), None)
            if key is None:
                continue

            df = pd.read_csv(file)

            results[folder.name][key] = {
                "given_numbers": df["given_number"].tolist(),
                "estimates": df["estimation"].tolist(),
            }

    return dict(results) 
