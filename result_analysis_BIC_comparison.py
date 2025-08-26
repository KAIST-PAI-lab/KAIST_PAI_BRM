# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize

import simulation_NLE.psychometric_functions as psy_funcs


def compute_bic(optimized_result, num_data_points):
    """
    optimized_result: scipy.optimize.minimize object
    num_data_points: number of data points (len(simulated_estimate))
    """
    k = len(optimized_result.x)
    print(k)
    nll = optimized_result.fun
    bic = k * np.log(num_data_points) + 2 * nll

    return bic


# load config
sim_config_path = sim_config_path = (
    Path(__file__).parent / "simulation_NLE" / "simulation_config.yaml"
)
with sim_config_path.open(encoding="utf-8") as f:
    SIM_CONFIG = yaml.safe_load(f)


N_MAX = SIM_CONFIG["parameter_lists"]["N_MAX"]
N_TRIALS = SIM_CONFIG["parameter_lists"]["N_TRIALS"]
GRID_RESOLUTION = SIM_CONFIG["parameter_lists"]["GRID_RESOLUTION"]

plots_save_path = "plots"
data_dir = Path("data")
participants_codes = []
bic_diff_gpal = []
bic_diff_ado = []
bic_diff_bd = []
for folder in data_dir.iterdir():
    if folder.is_dir():
        print(f"Current Folder: {folder.name}")
        for file in folder.iterdir():
            if file.is_file():
                print(f"Current File: {file.name}")
                file_name = file.name
                if (
                    "ado_results" in file_name
                    or "gpal_results" in file_name
                    or "random_results" in file_name
                ) and file.suffix == ".csv":

                    df = pd.read_csv(file)
                    given_numbers = df["given_number"].tolist()
                    estimates = df["estimation"].tolist()

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
                        args=(np.array(given_numbers), np.array(estimates)),
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
                        args=(np.array(given_numbers), np.array(estimates)),
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
                        args=(np.array(given_numbers), np.array(estimates)),
                        method="L-BFGS-B",
                    )

                    x_grid = np.linspace(1, N_MAX, GRID_RESOLUTION)

                    mllm_func = psy_funcs.mixed_log_linear
                    one_cpm_func = psy_funcs.one_cyclic_power
                    two_cpm_func = psy_funcs.two_cyclic_power
                    mllm_params_opt = optimized_mixed_log_linear.x
                    one_cpm_params_opt = optimized_one_cyclic_power.x
                    two_cpm_params_opt = optimized_two_cyclic_power.x

                    mllm_params_for_curve = mllm_params_opt[:-1]
                    one_cpm_params_for_curve = one_cpm_params_opt[:-1]
                    two_cpm_params_for_curve = two_cpm_params_opt[:-1]

                    y_mllm = mllm_func(*mllm_params_for_curve, given_number=x_grid)
                    y_onecpm = one_cpm_func(
                        *one_cpm_params_for_curve, given_number=x_grid
                    )
                    y_twocpm = [
                        two_cpm_func(*two_cpm_params_for_curve, given_number=x_value)
                        for x_value in x_grid
                    ]

                    MLLM_fitted_params_str = ""
                    for i in range(len(MLLM_model_param_names)):
                        param_name = MLLM_model_param_names[i][6:]
                        optimized_value = round(mllm_params_opt[i], 2)

                        if i == len(MLLM_model_param_names) - 1:
                            MLLM_fitted_params_str += f"{param_name}: {optimized_value}"

                        else:
                            MLLM_fitted_params_str += (
                                f"{param_name}: {optimized_value}, "
                            )

                    oneCPM_fitted_params_str = ""
                    for i in range(len(oneCPM_model_param_names)):
                        param_name = oneCPM_model_param_names[i][6:]
                        optimized_value = round(one_cpm_params_opt[i], 2)
                        if i == len(oneCPM_model_param_names) - 1:
                            oneCPM_fitted_params_str += (
                                f"{param_name}: {optimized_value}"
                            )
                        else:
                            oneCPM_fitted_params_str += (
                                f"{param_name}: {optimized_value}, "
                            )

                    twoCPM_fitted_params_str = ""
                    for i in range(len(twoCPM_model_param_names)):
                        param_name = twoCPM_model_param_names[i][6:]
                        optimized_value = round(two_cpm_params_opt[i], 2)
                        if i == len(twoCPM_model_param_names) - 1:
                            twoCPM_fitted_params_str += (
                                f"{param_name}: {optimized_value}"
                            )
                        else:
                            twoCPM_fitted_params_str += (
                                f"{param_name}: {optimized_value}, "
                            )

                    plt.figure(figsize=(7, 5))
                    plt.scatter(
                        given_numbers, estimates, alpha=0.4, label="Data points"
                    )
                    plt.plot(x_grid, y_mllm, label=f"MLLM ({MLLM_fitted_params_str})")
                    plt.plot(
                        x_grid, y_onecpm, label=f"1CPM ({oneCPM_fitted_params_str})"
                    )
                    plt.plot(
                        x_grid, y_twocpm, label=f"2CPM ({twoCPM_fitted_params_str})"
                    )
                    plt.title(file.name)
                    plt.xlabel("Given Number")
                    plt.ylabel("Estimate")
                    plt.grid(True)
                    plt.legend()
                    plt.tight_layout()
                    title = f"Models fitted - {file.name}"
                    plt.title(title)
                    plt.savefig(fname=f"plots/{title}.png")
                    plt.show()

                    # BIC comparison plots
                    bic_mllm = compute_bic(
                        optimized_mixed_log_linear, len(given_numbers)
                    )
                    bic_1cpm = compute_bic(
                        optimized_one_cyclic_power, len(given_numbers)
                    )
                    bic_2cpm = compute_bic(
                        optimized_two_cyclic_power, len(given_numbers)
                    )

                    bic_diff = bic_mllm - bic_1cpm

                    if "ado_results" in file_name:
                        bic_diff_ado.append(bic_diff)
                    elif "gpal_results" in file_name:
                        bic_diff_gpal.append(bic_diff)
                    elif "random_results" in file_name:
                        bic_diff_bd.append(bic_diff)

                    plt.figure(figsize=(6, 4))
                    bars = plt.bar(
                        ["MLLM", "1CPM", "2CPM"], [bic_mllm, bic_1cpm, bic_2cpm]
                    )
                    plt.ylabel("BIC")
                    title = f"BIC comparison - {file.name}"
                    plt.title(title)
                    plt.grid(axis="y", linestyle="--", alpha=0.5)

                    # label
                    for bar in bars:
                        h = bar.get_height()
                        plt.text(
                            bar.get_x() + bar.get_width() / 2,
                            h,
                            f"{h:.2f}",
                            ha="center",
                            va="bottom",
                            fontsize=10,
                        )
                    plt.tight_layout()
                    plt.savefig(fname=f"plots/{title}.png")
                    plt.show()

        participant_code = folder.name[-8:]
        participants_codes.append(participant_code)


x = np.arange(len(participants_codes))  # 0, 1, 2, ...
bar_width = 0.25

plt.figure(figsize=(10, 6))
plt.bar(x - bar_width, bic_diff_gpal, width=bar_width, label="GPAL")
plt.bar(x, bic_diff_ado, width=bar_width, label="ADO")
plt.bar(x + bar_width, bic_diff_bd, width=bar_width, label="BD")

plt.xlabel("Participant")
plt.ylabel("BIC Difference")
plt.title("BIC differences by optimization across participants")
plt.xticks(ticks=x, labels=participants_codes, rotation=45)
ymin, ymax = plt.ylim()

ymax = np.ceil(ymax / 5) * 5
ymin = np.floor(ymin / 5) * 5

plt.ylim(ymin, ymax)
plt.yticks(np.arange(ymin, ymax + 1, 5))
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)
for i in range(len(participants_codes) + 1):
    plt.axvline(i - 0.5, color="gray", linestyle="-", alpha=0.4)

plt.tight_layout()
title = "BIC differences by model across participants"
plt.savefig(fname=f"plots/{title}.png")
plt.show()

# %%

print("abc")
# %%
