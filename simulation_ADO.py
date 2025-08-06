# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize

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

grid_design = {"given_number": np.arange(1, N_MAX, 1)}

grid_param = {
    name: parse_parameter_range(conf)
    for name, conf in SIM_CONFIG["parameter_lists"]["params_ADO_" + assumed_model_name][
        "grid_parameters"
    ].items()
}
grid_response = {"simulated_estimate": np.arange(1, N_MAX, 1)}
print(grid_response)

engine = Engine(task, model, grid_design, grid_param, grid_response)

results = []

engine.reset()

print(true_params)

bic_list = []

for i in range(N_TRIALS):
    design = engine.get_design("optimal")

    response = true_model_simulate_response(
        **true_params, given_number=design["given_number"]
    )
    print(f"given number -> {design['given_number']} | response -> {response}")

    results.append([design["given_number"], response])
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
        x0.append(
            SIM_CONFIG["parameter_lists"]["params_optimize_" + assumed_model_name][
                name + "_initial"
            ]
        )

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

    plt.scatter(df["given_number"], df["estimate"], alpha=0.3)
    plt.plot(
        assumed_model(*optimized_parameters[:-1], np.linspace(0, N_MAX, N_MAX)),
        color="green",
    )

    plt.plot(
        true_model(**true_params_noiseless, given_number=np.linspace(0, N_MAX, N_MAX)),
        color="red",
    )

    plt.ylim([0, N_MAX])
    plt.xlabel("Given number", fontsize=15)
    plt.ylabel("Number estimate", fontsize=15)
    plt.legend(["Generated Data", "Fitted Function", "True Function"])
    plt.show()

    # BIC calculation
    BIC_log_linear = 3 * np.log(len(simulated_responses)) + 2 * optimized_model["fun"]
    print("BIC_log_linear", BIC_log_linear)
    print(compute_bic(optimized_model, len(simulated_responses)))

    bic = compute_bic(optimized_model, len(simulated_responses))

    bic_list.append(bic)

x = list(range(1, len(bic_list) + 1))
y = bic_list

plt.plot(x, y)
plt.xlabel("Trial")
plt.ylabel("BIC")
plt.title("BIC over Trials")
plt.grid(True)
plt.show()
