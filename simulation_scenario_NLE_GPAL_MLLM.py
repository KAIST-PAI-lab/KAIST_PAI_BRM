# %%
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.model_selection import train_test_split

import simulation_NLE.psychometric_functions as psy_funcs

sim_config_path = sim_config_path = (
    Path(__file__).parent / "simulation_NLE" / "simulation_config.yaml"
)
with sim_config_path.open(encoding="utf-8") as f:
    SIM_CONFIG = yaml.safe_load(f)

N_MAX = SIM_CONFIG["parameter_list"]["N_MAX"]


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

# %% Simulate one-trial response
cognitive_model_name = "mixed_log_linear"

true_params = SIM_CONFIG["parameter_lists"]["params_" + cognitive_model_name]
print(true_params)

# %%

parameter_list = [1, 0, 0.5]  # a,b,lambda
variance_scale = 5  # standard deviation of the random errors in response

given_number = 5

number_estimate = psy_funcs.mixed_log_linear_generate_response(
    **true_params, given_number=given_number
)

number_estimate = np.clip(number_estimate, 0, N_MAX)

given_number_list.append(given_number)
number_estimate_list.append(number_estimate)

# %% Define GP
# Define hyperparameters
error_variance = 1e-2  # response noise
length_scale = 10
output_variance = 10  # note that y will be normalized.

## RBF kernel
target_kernel = C(output_variance, constant_value_bounds="fixed") * RBF(
    length_scale=length_scale, length_scale_bounds="fixed"
)


# %%
# Fit GP with the selected kernel
X = np.array(given_number_list).reshape(-1, 1)
y = np.array(number_estimate_list)

gp = GaussianProcessRegressor(
    kernel=target_kernel, alpha=error_variance, normalize_y=True, n_restarts_optimizer=5
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
# %%
## uncertainty plot
plt.plot(y_std)

# %%
## find all elements with the maximum std
max_std_loc = np.argwhere(y_std == np.max(y_std))
## randomly select one element
next_design_loc = max_std_loc[np.random.choice(len(max_std_loc))]
## next design
next_design = np.squeeze(X_pred[next_design_loc])

plt.plot(y_std)
plt.scatter(next_design_loc, y_std[next_design_loc], color="red")

print("next given number =", next_design)

# %% Simulate the next trial
given_number = next_design
number_estimate = log_linear(given_number, parameter_list, N_MAX) + np.random.normal(
    loc=0, scale=variance_scale
)

given_number_list.append(given_number)
number_estimate_list.append(number_estimate)

# %%
# Fit GP with the selected kernel
X = np.array(given_number_list).reshape(-1, 1)
y = np.array(number_estimate_list)

gp = GaussianProcessRegressor(
    kernel=target_kernel, alpha=error_variance, normalize_y=True, n_restarts_optimizer=5
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


# %% Next trial
repetitions = 49
for _ in range(repetitions):
    ## find all elements with the maximum std
    max_std_loc = np.argwhere(y_std == np.max(y_std))
    ## randomly select one element
    next_design_loc = max_std_loc[np.random.choice(len(max_std_loc))]
    ## next design
    next_design = np.squeeze(X_pred[next_design_loc])

    plt.plot(y_std)
    plt.scatter(next_design_loc, y_std[next_design_loc], color="red")

    print("next given number =", next_design)

    given_number = next_design
    number_estimate = log_linear(
        given_number, parameter_list, N_MAX
    ) + np.random.normal(loc=0, scale=variance_scale)

    given_number_list.append(given_number)
    number_estimate_list.append(number_estimate)

    # Fit GP with the selected kernel
    X = np.array(given_number_list).reshape(-1, 1)
    y = np.array(number_estimate_list)

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

# %%
# %%
# %%
# %%
