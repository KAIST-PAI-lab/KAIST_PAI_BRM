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
        
    def respond(self, stimulus, params_override):
        if params_override:
            params = params_override
        else:
            params = self.true_model_params
            
        print(f"current params:", params)

        response = self.generate_response(**params, given_number=stimulus)

        print(f"Participant responds with {response} to the stimulus {stimulus} (w/ {params})")

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
    fig.tight_layout()


    return fig, ax, optimized_model

    # plot the datapoints and the fitted model in a figure   


def fit_model(data_points_x, data_points_y, fitting_model_name, manual_title = None):
    model_param_names = list(
        SIM_CONFIG["parameter_lists"]["params_" + fitting_model_name].keys()
    )

    like_fn = getattr(psy_funcs, fitting_model_name + "_likelihood_scipy_minimize")
    true_model = getattr(psy_funcs, fitting_model_name)

    x0 = []
    for name in model_param_names:
        start_value = SIM_CONFIG["parameter_lists"]["params_optimize_" + fitting_model_name][f"{name}_range_start"]
        end_value   = SIM_CONFIG["parameter_lists"]["params_optimize_" + fitting_model_name][f"{name}_range_end"]
        choices = np.round(np.arange(start_value, end_value, 0.1), 1)
        x0.append(float(np.random.choice(choices)))


    bounds = []
    for name in model_param_names:
        start_value = SIM_CONFIG["parameter_lists"]["params_optimize_" + fitting_model_name][f"{name}_range_start"]
        end_value   = SIM_CONFIG["parameter_lists"]["params_optimize_" + fitting_model_name][f"{name}_range_end"]
        bounds.append((start_value, end_value))

    data_points_x = np.asarray(data_points_x)
    data_points_y = np.asarray(data_points_y)

    print(f"Optimizing {fitting_model_name} to {len(data_points_x)} X and {len(data_points_y)} Y")
    print(f"x0: {x0}")
    print(f"bounds: {bounds}")


    optimized_model = minimize(
        fun=like_fn,
        x0=np.asarray(x0, dtype=float),
        bounds=bounds,
        args=(data_points_x, data_points_y),
        method="L-BFGS-B",
    )

    optimized_parameters = optimized_model["x"]

    x_range = np.arange(0, int(N_MAX) + 1)

    y_pred = true_model(*optimized_parameters[:-1], x_range)
    
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.scatter(data_points_x, data_points_y, s=25, alpha=0.8, label="Observed", color="black")
    ax.plot(x_range, y_pred, linewidth=2, label=f"{fitting_model_name}")

    param_str = ""

    for name, value in zip(model_param_names, optimized_parameters):
        param_str += f"{name[6:]}:{round(value,2)} "

    default_info = f"{fitting_model_name}\n{param_str}"
    if manual_title:
        title = manual_title + "\n" + default_info
    else:
        title = default_info

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig, ax, optimized_model


def get_GP_mean_function(data_points_x, data_points_y, gp_regressor, manual_title=None):
    data_points_x_reshaped = np.array(data_points_x).reshape(-1, 1)
    
    gp_regressor.fit(data_points_x_reshaped, data_points_y)

    x_range_pred = np.arange(0, N_MAX+1).reshape(-1, 1)

    gp_mean_function = gp_regressor.predict(x_range_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    default_info = "GP Mean Function"
    if manual_title:
        title = manual_title + "\n" + default_info
    else:
        title = default_info
    ax.set_title(title)
    ax.plot(x_range_pred.ravel(), gp_mean_function)
    ax.scatter(data_points_x, data_points_y, color="black")
    fig.tight_layout()

    return fig, ax, gp_regressor


def run_residual_analysis(data_points_x, data_points_y, model_name, parameters, manual_title = None):
    
    model = getattr(psy_funcs, model_name)

    predicted_values = [model(*parameters, x_value) for x_value in data_points_x]

    print(f"Model Predictions: {predicted_values}")

    residuals = [p_value - y_value for p_value, y_value in zip(predicted_values, data_points_y)]

    mean = np.mean(residuals)

    std= np.std(residuals)

    fig, ax = plt.subplots(figsize=(6, 4))

    # Residual histogram
    ax.hist(residuals, bins=20, color="skyblue", edgecolor="black", alpha=0.7)

    default_info = f"Residual Analysis\nmean={mean:.2f}, std={std:.2f}"
    if manual_title:
        title = manual_title + "\n" + default_info
    else:
        title = default_info
    
    ax.set_title(title)

    ax.set_xlabel("Residual")
    ax.set_ylabel("Frequency")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    return fig, ax, residuals


def run_residual_analysis_GP(data_points_x, data_points_y, gp_regressor, manual_title = None):
    
    data_points_x_reshaped = np.array(data_points_x).reshape(-1, 1)
    
    gp_regressor.fit(data_points_x_reshaped, data_points_y)

    gp_predictions = gp_regressor.predict(data_points_x_reshaped)
    
    residuals = [p_value - y_value for p_value, y_value in zip(gp_predictions, data_points_y)]

    fig, ax = plt.subplots(figsize=(6, 4))

    mean = np.mean(residuals)

    std= np.std(residuals)

    # Residual histogram
    ax.hist(residuals, bins=20, color="skyblue", edgecolor="black", alpha=0.7)
    
    default_info = f"Residuals Analysis (GP)\nmean={mean:.2f}, std={std:.2f}"
    if manual_title:
        title = manual_title + "\n" + default_info
    else:
        title = default_info
    
    ax.set_title(title)
    fig.tight_layout()
    ax.set_xlabel("Residual")
    ax.set_ylabel("Frequency")
    ax.grid(alpha=0.3)

    return fig, ax, gp_regressor



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
