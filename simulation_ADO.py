# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

import simulation_NLE.psychometric_functions as psy_funcs
from adopy import Engine, Model, Task

sim_config_path = sim_config_path = (
    Path(__file__).parent / "simulation_NLE" / "simulation_config.yaml"
)
with sim_config_path.open(encoding="utf-8") as f:
    SIM_CONFIG = yaml.safe_load(f)


def parse_parameter_range(param_conf):
    if param_conf["type"] == "linspace":
        return np.linspace(param_conf["start"], param_conf["end"], param_conf["num"])
    elif param_conf["type"] == "geomspace":
        return np.geomspace(param_conf["start"], param_conf["end"], param_conf["num"])
    elif param_conf["type"] == "arange":
        return np.arange(param_conf["start"], param_conf["end"], param_conf["num"])
    else:
        raise ValueError(f"Unknown param type: {param_conf['type']}")


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


print("Initiating ADO simulation")

N_MAX = SIM_CONFIG["parameter_lists"]["N_MAX"]
N_TRIALS = SIM_CONFIG["parameter_lists"]["N_TRIALS"]
GRID_RESOLUTION = SIM_CONFIG["parameter_lists"]["GRID_RESOLUTION"]

# assumed model objects
assumed_model_name = SIM_CONFIG["assumed_function"]

print(f"Assumed Model: {assumed_model_name}")


assumed_model = getattr(psy_funcs, assumed_model_name)

assumed_model_likelihood = getattr(psy_funcs, assumed_model_name + "_likelihood")

assumed_model_likelihood_scipy_minimize = getattr(
    psy_funcs, assumed_model_name + "_likelihood_scipy_minimize"
)

assumed_model_simulate_response = getattr(
    psy_funcs, assumed_model_name + "_generate_response"
)

# true model objects
true_model_name = SIM_CONFIG["true_function"]
print(f"True Model: {true_model_name}")

true_model = getattr(psy_funcs, true_model_name)

true_model_simulate_response = getattr(
    psy_funcs, true_model_name + "_generate_response"
)

true_model_likelihood_scipy_minimize = getattr(
    psy_funcs, true_model_name + "_likelihood_scipy_minimize"
)

true_params = SIM_CONFIG["parameter_lists"]["params_" + true_model_name]

# run ADO - initialize task, model, engine
task = Task(
    name="1-D Number Line Estimation",
    designs=["given_number"],
    responses=["simulated_estimate"],
)

model_param_names_list = list(
    SIM_CONFIG["parameter_lists"]["params_" + assumed_model_name].keys()
)
print(model_param_names_list)
print(assumed_model_likelihood)
model = Model(
    name=assumed_model_name,
    task=task,
    params=model_param_names_list,
    func=assumed_model_likelihood,
)

grid_design = {"given_number": np.linspace(5, N_MAX, GRID_RESOLUTION)}

grid_param = {
    name: parse_parameter_range(conf)
    for name, conf in SIM_CONFIG["parameter_lists"]["params_ADO_" + assumed_model_name][
        "grid_parameters"
    ].items()
}

grid_response = {"simulated_estimate": np.linspace(5, N_MAX, GRID_RESOLUTION)}
print(grid_response)

engine = Engine(task, model, grid_design, grid_param, grid_response)
results = []

x_range = np.linspace(0, N_MAX, N_MAX)
colors = ["red", "orange", "blue", "purple"]

engine.reset()

scale = 1

bic_list = []

mse_list = []

for i in tqdm(range(N_TRIALS)):
    design = engine.get_design("optimal")

    response = true_model_simulate_response(
        **true_params, given_number=design["given_number"]
    )
    print(f"given number -> {design['given_number']} | response -> {response}")

    results.append([design["given_number"] * scale, response * scale])
    engine.update(design, response)

    df = pd.DataFrame(results, columns=["given_number", "estimate"])
    true_params_noiseless = true_params.copy()
    true_params_noiseless.pop("param_noise")

    simulated_responses = np.array([_[1] for _ in results])

    given_numbers = np.array([_[0] for _ in results])

    model_param_names = list(
        SIM_CONFIG["parameter_lists"]["params_" + assumed_model_name].keys()
    )

    x0 = []
    for name in model_param_names:
        start_value = SIM_CONFIG["parameter_lists"][
            "params_optimize_" + assumed_model_name
        ][name + "_range_start"]

        end_value = SIM_CONFIG["parameter_lists"][
            "params_optimize_" + assumed_model_name
        ][name + "_range_end"]
        choices = np.round(np.arange(1, 20.1, 0.1), 1)

        # Randomly select one value
        random_value = np.random.choice(choices)

        x0.append(random_value)

    bounds = []
    for name in model_param_names:
        bounds.append(
            (
                SIM_CONFIG["parameter_lists"]["params_optimize_" + assumed_model_name][
                    name + "_range_start"
                ],
                SIM_CONFIG["parameter_lists"]["params_optimize_" + assumed_model_name][
                    name + "_range_end"
                ],
            )
        )

    optimized_model = minimize(
        fun=assumed_model_likelihood_scipy_minimize,
        x0=x0,
        bounds=bounds,
        args=(given_numbers, simulated_responses),
        method="L-BFGS-B",
    )

    optimized_parameters = optimized_model["x"]

    print(f"optimized_parameters {optimized_parameters} ({bounds})")

    plt.scatter(df["given_number"], df["estimate"], alpha=0.3, label="Generated Data")
    plt.plot(
        assumed_model(*optimized_parameters[:-1], np.linspace(0, N_MAX, N_MAX)),
        color="green",
        label="Fitted Function",
    )

    plt.plot(
        true_model(
            **true_params_noiseless,
            given_number=np.linspace(0, N_MAX, N_MAX),
        ),
        label="True Function",
        color="red",
    )

    plt.ylim([0, N_MAX])
    plt.xlabel("Given number", fontsize=15)
    plt.ylabel("Number estimate", fontsize=15)
    plt.legend(loc="best")
    plt.show()

    # BIC calculation
    bic = compute_bic(optimized_model, len(simulated_responses))

    bic_list.append(bic)

    # GP mean function estimation & the distance to the true function
    # Define hyperparameters
    error_variance = 1e-2
    length_scale = 10
    output_variance = 10

    target_kernel = C(output_variance) * RBF(length_scale=length_scale)

    x_train = [int(num) for num in given_numbers]
    x_train = np.array(x_train).reshape(-1, 1)

    y_train = [int(num) for num in simulated_responses]
    y_train = np.array(y_train)

    gp_estimation = GaussianProcessRegressor(
        kernel=target_kernel,
        alpha=error_variance,
        normalize_y=True,
        n_restarts_optimizer=100,
    )

    gp_estimation.fit(x_train, y_train)

    x_range_2d = np.linspace(1, N_MAX, N_MAX).reshape(-1, 1)

    gp_mean_function = gp_estimation.predict(x_range_2d)
    true_function_outputs = true_model(**true_params_noiseless, given_number=x_range)
    true_function_outputs = true_function_outputs.clip(0, N_MAX)

    mse = mean_squared_error(true_function_outputs, gp_mean_function)
    mse_list.append(mse)
    number_current_trial = len(mse_list)
    # GP mean function & true function visualized
    sigma = SIM_CONFIG["parameter_lists"]["params_" + true_model_name]["param_noise"]
    plt.figure(figsize=(6, 5))
    plt.plot(x_range, true_function_outputs, color=colors[0], label="True Function")
    plt.plot(x_range, gp_mean_function, color=colors[1], label="GP Mean Function")
    plt.title(f"MSE values over trials, sigma={sigma}")
    plt.xlabel("Given Number")
    plt.ylabel("Estimated Value")
    plt.title(f"N_TRIALS={len(mse_list)}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"GP_func_and_true_func_trial_{number_current_trial}.png")
    plt.show()

    # MSE values and number of trials
    plt.figure(figsize=(6, 5))
    plt.plot(range(1, number_current_trial + 1), mse_list, marker="o")
    plt.title(f"MSE values over trials, sigma={sigma}")
    plt.xlabel("N_TRIALS")
    plt.xticks(range(1, number_current_trial + 1))
    plt.ylabel("MSE")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# show BIC values across trials
# x = list(range(1, len(bic_list) + 1))
# y = bic_list

# plt.plot(x, y)
# plt.xlabel("Trial")
# plt.ylabel("BIC")
# plt.title("BIC over Trials")
# plt.grid(True)
# plt.show()

with sim_config_path.open(encoding="utf-8") as f:
    SIM_CONFIG = yaml.safe_load(f)
model_names = SIM_CONFIG["function_names"]

BICs = []

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
        start_value = SIM_CONFIG["parameter_lists"]["params_optimize_" + model_name][
            name + "_range_start"
        ]

        end_value = SIM_CONFIG["parameter_lists"]["params_optimize_" + model_name][
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
                SIM_CONFIG["parameter_lists"]["params_optimize_" + model_name][
                    name + "_range_start"
                ],
                SIM_CONFIG["parameter_lists"]["params_optimize_" + model_name][
                    name + "_range_end"
                ],
            )
        )

    simulated_responses = np.array([_[1] for _ in results])

    given_numbers = np.array([_[0] for _ in results])

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

    df = pd.DataFrame(results, columns=["given_number", "estimate"])
    y_pred = model_standard(*optimized_parameters[:-1], x_range)

    plt.scatter(df["given_number"], df["estimate"], alpha=0.3)
    plt.plot(x_range, y_pred, color=colors[n], label=f"{model_name}")
    plt.xlabel("Given Number")
    plt.ylabel("Number Estimate")
    plt.title("Models fitted to the simulated data")
    plt.grid(True)

y_pred = true_model(**true_params_noiseless, given_number=x_range)

plt.plot(x_range, y_pred, color=colors[-1], label="True Model")
plt.legend(loc="best")
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
        ha="center",
        va="bottom",
        fontsize=10,
    )
plt.show()
# %%
