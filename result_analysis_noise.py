#%%
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

plots_save_path = "plots"
data_dir = Path("data")
participants_codes = []

given_numbers_and_estimates = defaultdict(list)

for given_number, estimate in zip(given_numbers, estimates):
    given_numbers_and_estimates[given_number].append(estimate)

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
                    
                    for given_number, estimate in zip(given_numbers, estimates):
                        given_numbers_and_estimates[given_number].append(estimate)
                    

#%%
given_numbers_and_stdevs = defaultdict(list)

for given_number, estimates in sorted(given_numbers_and_estimates.items()):
    
    std = np.std(estimates, ddof=1)
    print(given_number, std)

    given_numbers_and_stdevs[given_number].append(std)


stdev_mean = np.nanmean(list(given_numbers_and_stdevs.values()))
print(f"Standard Deviation Mean: {stdev_mean}")
