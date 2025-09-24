#%%
import pickle
import glob
from pathlib import Path
from utils_simulation import *
from matplotlib.ticker import MultipleLocator
import re

def find_missing_checkpoints(checkpoint_folder: Path):
    numbers = []
    for file in checkpoint_folder.glob("*.pkl"):
        match = re.search(r"#(\d+)_", file.name)
        if match:
            numbers.append(int(match.group(1)))


    numbers = sorted(set(numbers))
    min_num, max_num = numbers[0], numbers[-1]

    missing = [i for i in range(min_num, max_num + 1) if i not in numbers]

    print(f"총 파일 개수: {len(numbers)}")
    print(f"번호 범위: {min_num} ~ {max_num}")
    if missing:
        print("누락된 번호:", missing)
    else:
        print("누락된 파일 없음")


simulation_number = 6
checkpoint_folder_BR = Path(f"simulation_checkpoints/GPAL_param_robustness_#{simulation_number}/records_BR")

checkpoint_folder_GPAL = Path(f"simulation_checkpoints/GPAL_param_robustness_#{simulation_number}/records_GPAL")


find_missing_checkpoints(checkpoint_folder_BR)
num_files_BR = len([f for f in checkpoint_folder_BR.iterdir() if f.is_file()])
num_files_GPAL = len([f for f in checkpoint_folder_GPAL.iterdir() if f.is_file()])

print(f"N_BR samples: {num_files_BR}")
print(f"N_GPAL samples: {num_files_GPAL}")

# true ~1.6
# estimate ~1.6

N_TRIALS = 50


mapping = {
    0: "mixed_log_linear",
    1: "one_cyclic_power",
    2: "two_cyclic_power",
    3: "one_cyclic_power",
    4: "two_cyclic_power",
    5: "one_cyclic_power",
    6: "two_cyclic_power",
}

true_func_name = mapping.get(simulation_number, "unknown")

# param_analysis_range = (0.75, 1) # slope when MLLM, exponent when CPM
param_analysis_range = (-999, 999) # slope when MLLM, exponent when CPM

true_param_values_BR = [[]for i in range(N_TRIALS)]
optimized_param_values_BR = [[]for i in range(N_TRIALS)]
pearson_values_BR = []
pearson_ci_low_BR = []
pearson_ci_high_BR = []

true_param_values_GPAL = [[]for i in range(N_TRIALS)]
optimized_param_values_GPAL  = [[]for i in range(N_TRIALS)]
pearson_values_GPAL = []
pearson_ci_low_GPAL = []
pearson_ci_high_GPAL = []

for pkl_file in checkpoint_folder_BR.glob("*.pkl"):
    with open(pkl_file, "rb") as f:
        checkpoint = pickle.load(f)
    for i in range(N_TRIALS):
        if simulation_number == 0:
            range_param = checkpoint["parameters"][0]
            true_param = checkpoint["parameters"][2]
        else:
            true_param = checkpoint["parameters"][0][0]
            range_param = true_param

        if param_analysis_range[0] <= range_param <= param_analysis_range[1]:
            true_param_values_BR[i].append(true_param)
            if simulation_number == 0:
                optimized_param = checkpoint["optimized_parameters"][i][2]
            else:
                optimized_param = checkpoint["optimized_parameters"][i][0]

            optimized_param_values_BR[i].append(optimized_param)

for pkl_file in checkpoint_folder_GPAL.glob("*.pkl"):
    with open(pkl_file, "rb") as f:
        checkpoint = pickle.load(f)
    for i in range(N_TRIALS):
        if simulation_number == 0:
            range_param = checkpoint["parameters"][0]
            true_param = checkpoint["parameters"][2]
        else:
            true_param = checkpoint["parameters"][0][0]
            range_param = true_param

        if param_analysis_range[0] <= range_param <= param_analysis_range[1]:
            true_param_values_GPAL[i].append(true_param)
            if simulation_number == 0:
                optimized_param = checkpoint["optimized_parameters"][i][2]
            else:
                optimized_param = checkpoint["optimized_parameters"][i][0]


            # data_points_x = checkpoint["stimuli"]
            # data_points_y = checkpoint["responses"]
            # print(true_param)
            # plot_data_points_and_function(data_points_x=data_points_x,
            #                               data_points_y=data_points_y,
            #                               function_name=true_func_name,
            #                               parameters=true_param,
            #                               manual_title=f"{true_param}")
            
            # break
            
            optimized_param_values_GPAL[i].append(optimized_param)

N_SAMPLES = len(optimized_param_values_BR[i])

for i in range(N_TRIALS):
    r_BR, ci_low_BR, ci_high_BR = compute_correlation_confidence_interval(true_param_values_BR[i], optimized_param_values_BR[i])
    r_GPAL, ci_low_GPAL, ci_high_GPAL = compute_correlation_confidence_interval(true_param_values_GPAL[i], optimized_param_values_GPAL[i])

    pearson_values_BR.append(r_BR)
    pearson_ci_low_BR.append(ci_low_BR)
    pearson_ci_high_BR.append(ci_high_BR)

    pearson_values_GPAL.append(r_GPAL)
    pearson_ci_low_GPAL.append(ci_low_GPAL)
    pearson_ci_high_GPAL.append(ci_high_GPAL)

figure, ax = plt.subplots(figsize = (7,4))
font_size_title = 25
font_size_legend = 15
font_size_axis_labels = 15
font_size_tick_size = 10
data_points_size = 50
data_points_opacity = 0.4
fill_between_opacity = 0.4

trials = range(1, N_TRIALS+1)

ax.plot(trials, pearson_values_GPAL, label="GPAL", linestyle="-")
ax.fill_between(trials, pearson_ci_low_GPAL, pearson_ci_high_GPAL, alpha=fill_between_opacity)

ax.plot(trials, pearson_values_BR, label="BR", linestyle="--")
ax.fill_between(trials, pearson_ci_low_BR, pearson_ci_high_BR, alpha=fill_between_opacity)

ax.set_xlim(-1, N_TRIALS+1)

ax.set_xlabel(f"Trial", fontsize=font_size_axis_labels)
ax.set_ylabel(f"Pearson's r", fontsize=font_size_axis_labels)
ax.set_ylim(0, 1.05)
ax = plt.gca()
ax.xaxis.set_major_locator(MultipleLocator(10))
ax.yaxis.set_major_locator(MultipleLocator(0.2))

ax.tick_params(axis="both", labelsize=font_size_tick_size)

ax.set_title(f"{true_func_name}\nSampling range={param_analysis_range}, N_SAMPLES={N_SAMPLES}")
ax.legend(fontsize = font_size_legend)
ax.grid(True)
figure.tight_layout()

# Task: True and Estimated scatter plot
figure, ax = plt.subplots(figsize = (6,4))

# true & estimated params scattered
for i in range(N_TRIALS):
    ax.scatter(true_param_values_BR[i], optimized_param_values_BR[i], color="tab:blue", alpha=0.4)

# x = true
# y = estimated
ax.set_title(f"{true_func_name} (BR)\nSampling range={param_analysis_range}, N_SAMPLES={N_SAMPLES}")
ax.set_xlabel("True Value")
ax.set_ylabel("Estimated Value")
figure.tight_layout()
figure.show()

# true & estimated params scattered
figure, ax = plt.subplots(figsize = (6,4))

for i in range(N_TRIALS):
    ax.scatter(true_param_values_GPAL[i], optimized_param_values_GPAL[i], color="tab:blue", alpha=0.4)

# x = true
# y = estimated
ax.set_title(f"{true_func_name} (GPAL)\nSampling range={param_analysis_range}, N_SAMPLES={N_SAMPLES}")
ax.set_xlabel("True Value")
ax.set_ylabel("Estimated Value")
figure.tight_layout()
figure.show()

# %%
