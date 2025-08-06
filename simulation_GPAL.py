# %%
import random
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel as C
from tqdm import tqdm

import simulation_NLE.psychometric_functions as psy_funcs

print("Initiating GPAL simulation")

sim_config_path = sim_config_path = (
    Path(__file__).parent / "simulation_NLE" / "simulation_config.yaml"
)
with sim_config_path.open(encoding="utf-8") as f:
    SIM_CONFIG = yaml.safe_load(f)

N_MAX = SIM_CONFIG["parameter_lists"]["N_MAX"]
N_TRIALS = SIM_CONFIG["parameter_lists"]["N_TRIALS"]
GRID_RESOLUTION = SIM_CONFIG["parameter_lists"]["GRID_RESOLUTION"]

# true model objects
true_model_name = SIM_CONFIG["true_function"]
print(f"True Model: {true_model_name}")

true_params = SIM_CONFIG["parameter_lists"]["params_" + true_model_name]

true_model = getattr(psy_funcs, true_model_name)

true_model_likelihood_scipy_minimize = getattr(
    psy_funcs, true_model_name + "_likelihood_scipy_minimize"
)

true_model_simulate_response = getattr(
    psy_funcs, true_model_name + "_generate_response"
)


def compute_bic(optimized_result, num_data_points):
    """
    optimized_result: scipy.optimize.minimize object
    num_data_points: number of data points (len(simulated_estimate))
    """
    k = len(optimized_result.x)  # 추정한 파라미터 개수
    print(k)
    nll = optimized_result.fun  # 최소화된 negative log likelihood
    bic = k * np.log(num_data_points) + 2 * nll

    return bic


warnings.filterwarnings("ignore")
given_number_list = []
number_estimate_list = []
X_pred = np.arange(1, N_MAX).reshape(-1, 1)

true_params = SIM_CONFIG["parameter_lists"]["params_" + true_model_name]
print(true_params)

given_number_list = []
simulated_response_list = []
bic_list = []

given_number_first = random.randint(1, N_MAX)
given_number_list.append(given_number_first)

for _ in tqdm(range(N_TRIALS)):
    simulated_response = true_model_simulate_response(
        **true_params, given_number=given_number_list[-1]
    )

    # simulated_response = np.clip(simulated_response, 0, N_MAX)

    simulated_response_list.append(simulated_response)

    # Define GP
    # Define hyperparameters
    error_variance = 1e-2
    length_scale = 10
    output_variance = 10

    ## RBF kernel
    target_kernel = C(output_variance, constant_value_bounds="fixed") * RBF(
        length_scale=length_scale, length_scale_bounds="fixed"
    )
    # Fit GP with the selected kernel
    X = np.array(given_number_list).reshape(-1, 1)
    print(X)
    y = np.array(simulated_response_list)
    print(y)

    gp = GaussianProcessRegressor(
        kernel=target_kernel,
        alpha=error_variance,
        normalize_y=True,
        n_restarts_optimizer=5,
    )

    gp.fit(X, y)
    log_marginal_likelihood = gp.log_marginal_likelihood()

    y_pred, y_std = gp.predict(X_pred, return_std=True)

    # Plot
    plt.figure(figsize=(6, 4))
    plt.scatter(X, y, c="black", label="Data")
    plt.plot(X_pred, y_pred, label="Prediction")
    plt.fill_between(
        X_pred.ravel(), y_pred - y_std, y_pred + y_std, alpha=0.3, label="Uncertainty"
    )
    plt.xlabel("Given Number")
    plt.ylabel("Number Estimate")
    plt.legend()
    plt.tight_layout()
    plt.show()
    ## uncertainty plot

    ## find all elements with the maximum std
    max_std_loc = np.argwhere(y_std == np.max(y_std))
    ## randomly select one element
    next_design_loc = max_std_loc[np.random.choice(len(max_std_loc))]
    ## next design
    next_design = np.squeeze(X_pred[next_design_loc])

    plt.plot(y_std)
    plt.scatter(next_design_loc, y_std[next_design_loc], color="red")
    plt.show()

    model_param_names = list(
        SIM_CONFIG["parameter_lists"]["params_" + true_model_name].keys()
    )

    x0 = []
    for name in model_param_names:
        start_value = SIM_CONFIG["parameter_lists"][
            "params_optimize_" + true_model_name
        ][name + "_range_start"]

        end_value = SIM_CONFIG["parameter_lists"]["params_optimize_" + true_model_name][
            name + "_range_end"
        ]
        choices = np.round(np.arange(1, 20.1, 0.1), 1)

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

    print(f"x0: {x0}")
    print(f"bounds: {bounds}")

    optimized_model = minimize(
        fun=true_model_likelihood_scipy_minimize,
        x0=x0,
        bounds=bounds,
        args=(np.array(given_number_list), np.array(simulated_response_list)),
        method="L-BFGS-B",
    )

    optimized_parameters = optimized_model["x"]

    print(optimized_parameters)

    true_params_noiseless = true_params.copy()
    true_params_noiseless.pop("param_noise")

    plt.scatter(
        given_number_list,
        simulated_response_list,
        alpha=0.3,
        label=f"Simulated Data Points (n={len(simulated_response_list)})",
    )

    plt.plot(
        np.linspace(5, N_MAX, GRID_RESOLUTION),
        true_model(
            *optimized_parameters[:-1],
            given_number=np.linspace(0, N_MAX, GRID_RESOLUTION),
        ),
        color="green",
        label="Fitted Function",
    )

    plt.plot(
        np.linspace(5, N_MAX, GRID_RESOLUTION),
        true_model(
            **true_params_noiseless, given_number=np.linspace(0, N_MAX, GRID_RESOLUTION)
        ),
        color="red",
        label="True Function",
    )
    plt.xlabel("Given Number", fontsize=15)
    plt.ylabel("Number Estimate", fontsize=15)
    plt.legend(loc="best")
    plt.grid(True)

    plt.show()

    bic = compute_bic(optimized_model, len(simulated_response_list))

    bic_list.append(bic)

    print("next given number =", next_design)
    given_number_list.append(next_design)


x = list(range(1, len(bic_list) + 1))
y = bic_list

plt.plot(x, y)
plt.xlabel("Trial")
plt.ylabel("BIC")
plt.title("BIC over Trials")
plt.grid(True)
plt.show()

with sim_config_path.open(encoding="utf-8") as f:
    SIM_CONFIG = yaml.safe_load(f)
model_names = SIM_CONFIG["function_names"]

BICs = []

colors = ["red", "orange", "blue","purple"]

x_range = np.linspace(0, N_MAX, N_MAX)
for n, model_name in enumerate(model_names):
    print(model_name)
    model_standard = getattr(psy_funcs, model_name)
    model_likelihood_scipy_minimize = getattr(
        psy_funcs, model_name + "_likelihood_scipy_minimize"
    )

    model_param_names = list(
        SIM_CONFIG["parameter_lists"]["params_" + model_name].keys()
    )

    x0 = []
    for name in model_param_names:
        x0.append(
            SIM_CONFIG["parameter_lists"]["params_optimize_" + model_name][
                name + "_initial"
            ]
        )

    bounds = []
    for name in model_param_names:
        bounds.append(
            (
                SIM_CONFIG["parameter_lists"]["params_optimize_" + model_name][
                    name + "_range_start"
                ],
                SIM_CONFIG["parameter_lists"]["params_optimize_" + model_name][
                    name + "_range_end"
                ],
            )
        )

    simulated_responses = np.array(simulated_response_list)

    given_numbers = np.array(given_number_list[:N_TRIALS])

    optimized_model = minimize(
        fun=model_likelihood_scipy_minimize,
        x0=x0,
        bounds=bounds,
        args=(given_numbers, simulated_responses),
        method="L-BFGS-B",
    )

    optimized_parameters = optimized_model["x"]

    bic = compute_bic(optimized_model, len(simulated_responses))

    BICs.append(bic)

    y_pred = model_standard(*optimized_parameters[:-1], x_range)

    plt.plot(x_range, y_pred, color=colors[n], label=f"{model_name}")
    plt.legend(loc="best")
    plt.xlabel("Given Number")
    plt.ylabel("Estimate")
    plt.title("Models fitted to the simulated data")
    plt.grid(True)

y_pred = true_model(**true_params_noiseless, x_range)

plt.plot(x_range, y_pred, color=colors[-1], label="True Model")

plt.scatter(
    x=given_numbers, y=simulated_responses, alpha=0.3, label="Simulated Data Points"
)
plt.show()

# BIC plot
plt.figure()

bars = plt.bar(model_names, BICs)
plt.xlabel("Model Name")
plt.ylabel("BIC Value")
plt.title("BIC Comparison Between Models")
plt.xticks(rotation=45, ha="right")

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2, 
        height,                           
        f"{height:.2f}",                  
        ha="center", va="bottom",         
        fontsize=10
    )
