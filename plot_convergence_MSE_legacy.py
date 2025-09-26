# %%

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as t_dist
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# gp regressor
global x_range
x_range = np.arange(5, 501, 1)
y_range = np.arange(5, 501, 0.5)

given_numbers_ado = []
given_numbers_gpal = []
given_numbers_bd = []

estimates_ado = []
estimates_gpal = []
estimates_bd = []

# Iterate the files in the data folder
data_dir = Path("experiment_results")
subject_count = 0
for folder in data_dir.iterdir():
    subject_count += 1
    if folder.is_dir():
        print(f"subject count: {subject_count}")
        print(f"Current Folder: {folder.name}")
        participant_code = folder.name[-8:]
        print(f"Subject #{participant_code}")
        for file in folder.iterdir():
            if file.is_file():
                file_name = file.name
                if (
                    "ado_results" in file_name
                    or "gpal_results" in file_name
                    or "random_results" in file_name
                ) and file.suffix == ".csv":
                    print(f"Current File: {file.name}")
                    df = pd.read_csv(file)
                    given_numbers = df["given_number"].tolist()
                    estimates = df["estimation"].tolist()
                    if "ado_results" in file_name:
                        given_numbers_ado.append(given_numbers)
                        estimates_ado.append(estimates)
                    elif "gpal_results" in file_name:
                        given_numbers_gpal.append(given_numbers)
                        estimates_gpal.append(estimates)
                    elif "random_results" in file_name:
                        given_numbers_bd.append(given_numbers)
                        estimates_bd.append(estimates)


def get_convergence_mse(x_data_points, y_data_points):
    x_range_reshaped_for_gpr = np.array(x_range).reshape(-1, 1)
    num_data_points = len(x_data_points)

    gp_mean_function_list = []

    kernel = C(1.0) * RBF(length_scale=2.0) + WhiteKernel(
        noise_level=0.05, noise_level_bounds=(1e-2, 1e5)
    )

    gp_regressor = GaussianProcessRegressor(
        kernel=kernel, normalize_y=True, n_restarts_optimizer=100
    )

    for i in range(1, num_data_points + 1):
        x_data_points_current_trial = x_data_points[:i]
        x_data_points_reshaped_for_gpr = np.array(x_data_points_current_trial).reshape(
            -1, 1
        )

        y_data_points_current_trial = y_data_points[:i]
        y_data_points_reshaped_for_gpr = np.array(y_data_points_current_trial)

        gp_regressor.fit(x_data_points_reshaped_for_gpr, y_data_points_reshaped_for_gpr)

        gp_mean_function = gp_regressor.predict(x_range_reshaped_for_gpr)

        gp_mean_function_list.append(gp_mean_function)

    print(gp_regressor.kernel_)
    mse_values = []
    for i in range(num_data_points):
        mse = mean_squared_error(gp_mean_function_list[-1], gp_mean_function_list[i])
        mse_values.append(mse)

    return mse_values


def get_confidence_interval_t(values, confidence: float = 0.95):
    x = np.asarray(values, dtype=float)
    n = x.size
    if n == 0:
        raise ValueError("No valid data points.")
    mean = float(np.mean(x))
    if n == 1:
        return mean, mean, mean, 0.0, 1

    sd = float(np.std(x, ddof=1))
    se = sd / math.sqrt(n)

    alpha = 1.0 - confidence

    crit = float(t_dist.ppf(1 - alpha / 2, df=n - 1))

    lo = mean - crit * se
    hi = mean + crit * se
    return mean, lo, hi, se, n


mse_values_ado = []
mse_values_gpal = []
mse_values_bd = []

num_subjects = subject_count
for i in tqdm(range(num_subjects)):
    mse_values_ado.append(get_convergence_mse(given_numbers_ado[i], estimates_ado[i]))
    mse_values_gpal.append(
        get_convergence_mse(given_numbers_gpal[i], estimates_gpal[i])
    )
    mse_values_bd.append(get_convergence_mse(given_numbers_bd[i], estimates_bd[i]))

# %%

num_trials = 20

mse_mean_values_ado = []
mse_mean_values_gpal = []
mse_mean_values_bd = []

confidence_interval_values_ado = []
confidence_interval_values_gpal = []
confidence_interval_values_bd = []

for i in tqdm(range(num_trials)):
    values_per_trial_ado = []
    values_per_trial_gpal = []
    values_per_trial_bd = []
    for ii in range(num_subjects):
        values_per_trial_ado.append(mse_values_ado[ii][i])
        values_per_trial_gpal.append(mse_values_gpal[ii][i])
        values_per_trial_bd.append(mse_values_bd[ii][i])

    mse_mean_value_ado = round(sum(values_per_trial_ado) / len(values_per_trial_ado), 2)
    mse_mean_value_gpal = round(
        sum(values_per_trial_gpal) / len(values_per_trial_gpal), 2
    )
    mse_mean_value_bd = round(sum(values_per_trial_bd) / len(values_per_trial_bd), 2)

    mse_mean_values_ado.append(mse_mean_value_ado)
    mse_mean_values_gpal.append(mse_mean_value_gpal)
    mse_mean_values_bd.append(mse_mean_value_bd)

    confidence_interval_value_ado = get_confidence_interval_t(values_per_trial_ado)
    confidence_interval_value_gpal = get_confidence_interval_t(values_per_trial_gpal)
    confidence_interval_value_bd = get_confidence_interval_t(values_per_trial_bd)

    confidence_interval_values_ado.append(confidence_interval_value_ado)
    confidence_interval_values_gpal.append(confidence_interval_value_gpal)
    confidence_interval_values_bd.append(confidence_interval_value_bd)

# %%

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

trials = np.arange(1, 21)
figure, axes = plt.subplots(3, 1, figsize=(6, 9))
figure.subplots_adjust(hspace=0.5)

# --- ADO plot ---
axes[0].plot(
    trials,
    mse_mean_values_ado,
    marker="o",
    linewidth=1,
    markersize=3,
    label="Mean MSE value",
)
ado_confidence_high_values = [_[2] for _ in confidence_interval_values_ado]
ado_confidence_low_values = [_[1] for _ in confidence_interval_values_ado]
axes[0].fill_between(
    trials,
    ado_confidence_high_values,
    ado_confidence_low_values,
    alpha=0.75,
    color="skyblue",
    label="Conf. Interval (95%, double-tailed)",
)
axes[0].xaxis.set_major_locator(MultipleLocator(1))
axes[0].yaxis.set_major_locator(MultipleLocator(5000))
axes[0].set_title("ADO")
axes[0].legend()

# --- GPAL plot ---
axes[1].plot(trials, mse_mean_values_gpal, marker="o", linewidth=1, markersize=3)
gpal_confidence_high_values = [_[2] for _ in confidence_interval_values_gpal]
gpal_confidence_low_values = [_[1] for _ in confidence_interval_values_gpal]
axes[1].fill_between(
    trials,
    gpal_confidence_high_values,
    gpal_confidence_low_values,
    alpha=0.75,
    color="skyblue",
)
axes[1].xaxis.set_major_locator(MultipleLocator(1))
axes[1].yaxis.set_major_locator(MultipleLocator(5000))
axes[1].set_title("GPAL")
axes[1].legend()

# --- BD plot ---
axes[2].plot(trials, mse_mean_values_bd, marker="o", linewidth=1, markersize=3)
bd_confidence_high_values = [_[2] for _ in confidence_interval_values_bd]
bd_confidence_low_values = [_[1] for _ in confidence_interval_values_bd]
axes[2].fill_between(
    trials,
    bd_confidence_high_values,
    bd_confidence_low_values,
    alpha=0.75,
    color="skyblue",
)
axes[2].xaxis.set_major_locator(MultipleLocator(1))
axes[2].yaxis.set_major_locator(MultipleLocator(5000))
axes[2].set_title("BD")
axes[2].legend()

label_font_size = 10
axes[0].set_xlabel("Trials", fontsize=label_font_size)
axes[0].set_ylabel("MSE", fontsize=label_font_size)

axes[1].set_xlabel("Trials", fontsize=label_font_size)
axes[1].set_ylabel("MSE", fontsize=label_font_size)

axes[2].set_xlabel("Trials", fontsize=label_font_size)
axes[2].set_ylabel("MSE", fontsize=label_font_size)

for ax in axes:
    ax.tick_params(labelsize=7)


figure.suptitle("Mean Convergence Plot", fontsize=14, fontweight="bold")

plt.show()

# %%
