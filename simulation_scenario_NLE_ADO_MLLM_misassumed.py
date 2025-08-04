# Simulation scenario: run NLE with ADO, true: simple linear, asssume MLLM

# %%
from pathlib import Path

import numpy as np
import yaml

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


N_MAX = SIM_CONFIG["parameter_lists"]["N_MAX"]
N_TRIALS = SIM_CONFIG["parameter_lists"]["N_TRIALS"]

cognitive_model_name = SIM_CONFIG["psychometric_function"]

print(f"Selected Assumed Cognitive Model: {cognitive_model_name}")

assumed_cognitive_model = getattr(psy_funcs, cognitive_model_name)


def true_cognitive_model_generate_response(a, b, sigma, x):
    """
    y = a · exp(b·x)  +  (가우시안 노이즈)

    Parameters
    ----------
    a : float            # 스케일(초기값)
    b : float            # 성장률(growth rate)
    sigma : float        # 관측 노이즈 표준편차
    x : float or array   # 입력 값
    """
    pred = a * np.exp(b * x)  # 이상적 예측치
    response = np.random.normal(pred, sigma)  # 노이즈 추가
    response = np.clip(response, 0, N_MAX)  # 범위 제한(선택)
    return response


def true_cognitive_model(a, b, x):
    """
    y = a · exp(b·x)  (노이즈 없음, 이상적 곡선)

    Parameters
    ----------
    a : float
    b : float
    x : float or array
    """
    return a * np.exp(b * x)


cognitive_model_likelihood = getattr(psy_funcs, cognitive_model_name + "_likelihood")

cognitive_model_likelihood_scipy_minimize = getattr(
    psy_funcs, cognitive_model_name + "_likelihood_scipy_minimize"
)
cognitive_model_simulate_response = getattr(
    psy_funcs, cognitive_model_name + "_generate_response"
)

# %%

# run ADO & show results
task = Task(
    name="1-D Number Line Estimation",
    designs=["given_number"],
    responses=["simulated_estimate"],
)

model_param_names_list = list(
    SIM_CONFIG["parameter_lists"]["params_" + cognitive_model_name].keys()
)
print(model_param_names_list)
print(cognitive_model_likelihood)
model = Model(
    name=cognitive_model_name,
    task=task,
    params=model_param_names_list,
    func=cognitive_model_likelihood,
)

# %%

grid_design = {"given_number": np.arange(1, N_MAX, 1)}

grid_param = {
    name: parse_parameter_range(conf)
    for name, conf in SIM_CONFIG["parameter_lists"][
        "params_ADO_" + cognitive_model_name
    ]["grid_parameters"].items()
}
grid_response = {"simulated_estimate": np.arange(1, N_MAX, 1)}
print(grid_response)


# Initialize ADO Engine
engine = Engine(task, model, grid_design, grid_param, grid_response)
# %%

results = []

engine.reset()

# %%
true_params = [5, 0.1, 10]  # a,b, sigma

for i in range(N_TRIALS):
    design = engine.get_design("optimal")
    x = int(design["given_number"])
    print(x)
    response = true_cognitive_model_generate_response(*true_params, x=x)
    print(f"given number -> {design['given_number']} | response -> {response}")

    results.append([design["given_number"], response])
    engine.update(design, response)


# %%
import matplotlib.pyplot as plt
import pandas as pd

df = pd.DataFrame(results, columns=["given_number", "estimate"])

plt.scatter(df["given_number"], df["estimate"], alpha=0.3)
plt.plot(
    np.linspace(0, N_MAX, N_MAX),
    true_cognitive_model(*true_params[:2], x=np.linspace(0, N_MAX, N_MAX)),
    color="red",
)
plt.show()

# %%
from scipy.optimize import minimize

simulated_responses = np.array([_[1] for _ in results])

given_numbers = np.array([_[0] for _ in results])

optimized_model = minimize(
    fun=cognitive_model_likelihood_scipy_minimize,
    x0=np.array([0.5, 0, 0.5, 10]),
    bounds=((0, 1), (0, N_MAX), (0, 1), (0, 50)),  # TODO automize
    args=(given_numbers, simulated_responses),
    method="L-BFGS-B",
)

optimized_parameters = optimized_model["x"]

print(optimized_parameters)

# %% estimated function (model fitting) and visualize

plt.scatter(df["given_number"], df["estimate"], alpha=0.3)
plt.plot(
    np.linspace(0, N_MAX, N_MAX),
    assumed_cognitive_model(
        *optimized_parameters[:3], given_number=np.linspace(0, N_MAX, N_MAX)
    ),
    color="green",
)
plt.plot(
    np.linspace(0, N_MAX, N_MAX),
    true_cognitive_model(*true_params[:2], x=np.linspace(0, N_MAX, N_MAX)),
    color="red",
)
plt.xlabel("Given number", fontsize=15)
plt.ylabel("Number estimate", fontsize=15)
plt.legend(["Generated Data", "Estimated Function", "True Function"])
plt.show()
# %% run BIC

# BIC calculation
BIC_log_linear = 3 * np.log(len(simulated_responses)) + 2 * optimized_model["fun"]
print("BIC_log_linear", BIC_log_linear)


# %%
print(compute_bic(optimized_model, len(simulated_responses)))

# %%
