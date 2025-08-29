import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import multivariate_normal, norm
from scipy.spatial.distance import jensenshannon
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def get_jsd_array(x_data_array:npt.NDArray, y_data_array:npt.NDArray, num_trials:int):
    
    x_data_2D=x_data_array.reshape(-1,1)
    x_data_predict=np.linspace(5, 500, (500-5)//5+1).reshape(-1,1)
    kernel=C(1.0)*RBF(1.0)+WhiteKernel(0.05, (1e-2, 1e5))

    post_mean_array=np.zeros((num_trials, len(x_data_predict)))
    #post_cov_array=np.zeros((num_trials, len(x_data_predict), len(x_data_predict)))
    post_std_array=np.zeros((num_trials, len(x_data_predict)))
    for idx in range(1, num_trials+1):
        gpr=GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=25)
        x_collected=x_data_2D[:idx]
        y_collected=y_data_array[:idx]
        gpr.fit(x_collected, y_collected)
        post_mean, post_std=gpr.predict(x_data_predict, return_std=True)
        post_mean_array[idx-1]=post_mean
        post_std_array[idx-1]=post_std

    #ref_dist=multivariate_normal(mean=post_mean_array[-1], cov=post_cov_array[-1])

    fine_grid=np.arange(5, 500.5, 0.5)
    ref_pdf=np.zeros_like(fine_grid)
    for mean, std in zip(post_mean_array[-1], post_std_array[-1]):
        ref_pdf+=norm.pdf(fine_grid, mean, std)
        ref_pdf = ref_pdf/np.sum(ref_pdf) 
    jsd_array=np.zeros(num_trials,)
    
    for tIdx in range(num_trials):
        #trial_dist=multivariate_normal(mean=post_mean_array[tIdx], cov=post_cov_array[tIdx])
        #ref_probs=ref_dist.pdf(fine_grid)
        #trial_probs=trial_dist.pdf(fine_grid)

        trial_pdf=np.zeros_like(fine_grid)
        for mean, std in zip(post_mean_array[tIdx], post_std_array[tIdx]):
            trial_pdf+=norm.pdf(fine_grid, mean, std)
        trial_pdf = trial_pdf/np.sum(trial_pdf)

        jsd=np.square(jensenshannon(trial_pdf, ref_pdf))
        '''
        kl_ref=(ref_pdf*np.log(ref_pdf/(mean_pdf+1e-10)+1e-10))
        kl_1=np.sum(kl_ref[~np.isnan(kl_ref)])
        #trial_nonzero=np.nonzero(trial_pdf)
        kl_trial=trial_pdf*np.log(trial_pdf/(mean_pdf+1e-10)+1e-10)
        kl_2=np.sum(kl_trial[~np.isnan(kl_trial)])
        jsd=(kl_1+kl_2)/2
        jsd_array[tIdx]=jsd
        '''
        jsd_array[tIdx]=jsd
    return jsd_array



if __name__=="__main__":

    num_subjects=18
    num_trials=20

    given_numbers_ado = np.zeros((num_subjects, num_trials))
    given_numbers_gpal = np.zeros((num_subjects, num_trials))
    given_numbers_bd = np.zeros((num_subjects, num_trials))

    estimates_ado = np.zeros((num_subjects, num_trials))
    estimates_gpal = np.zeros((num_subjects, num_trials))
    estimates_bd = np.zeros((num_subjects, num_trials))


    # Iterate the files in the data folder
    data_dir = Path("experiment_results")
    subject_count = 0
    for sIdx, folder in enumerate(data_dir.iterdir()):
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
                        given_numbers = df["given_number"].to_numpy()
                        estimates = df["estimation"].to_numpy()
                        if "ado_results" in file_name:
                            given_numbers_ado[sIdx]=given_numbers
                            estimates_ado[sIdx]=estimates
                        elif "gpal_results" in file_name:
                            given_numbers_gpal[sIdx]=given_numbers
                            estimates_gpal[sIdx]=estimates
                        elif "random_results" in file_name:
                            given_numbers_bd[sIdx]=given_numbers
                            estimates_bd[sIdx]=estimates

    jsd_ado=np.zeros((num_subjects, num_trials))
    jsd_gpal=np.zeros((num_subjects, num_trials))
    jsd_bd=np.zeros((num_subjects, num_trials))
    jsd=np.stack([jsd_ado, jsd_gpal, jsd_bd], axis=0)
    
    for sIdx in tqdm(range(num_subjects)):
        jsd[0][sIdx]=get_jsd_array(given_numbers_ado[sIdx], estimates_ado[sIdx], num_trials)
        jsd[1][sIdx]=get_jsd_array(given_numbers_gpal[sIdx], estimates_gpal[sIdx], num_trials)
        jsd[2][sIdx]=get_jsd_array(given_numbers_bd[sIdx], estimates_gpal[sIdx], num_trials)

    jsd_mean=np.mean(jsd, axis=1)
    np.savetxt("jse_mean.csv", jsd_mean, delimiter=',')
    trials = np.arange(1, 21)
    figure, axes = plt.subplots(3, 1, figsize=(6, 9))
    figure.subplots_adjust(hspace=0.5)

    # --- ADO plot ---
    axes[0].plot(
        trials,
        jsd_mean[0],
        marker="o",
        linewidth=1,
        markersize=3,
        label="Mean JS divergence value",
    )
    axes[0].xaxis.set_major_locator(MultipleLocator(1))
    axes[0].yaxis.set_major_locator(MultipleLocator(5000))
    axes[0].set_title("ADO")
    axes[0].legend()

    # --- GPAL plot ---
    axes[1].plot(trials, jsd_mean[1], marker="o", linewidth=1, markersize=3)
    axes[1].xaxis.set_major_locator(MultipleLocator(1))
    axes[1].yaxis.set_major_locator(MultipleLocator(5000))
    axes[1].set_title("GPAL")
    axes[1].legend()

    # --- BD plot ---
    axes[2].plot(trials, jsd_mean[2], marker="o", linewidth=1, markersize=3)
    axes[2].xaxis.set_major_locator(MultipleLocator(1))
    axes[2].yaxis.set_major_locator(MultipleLocator(5000))
    axes[2].set_title("BD")
    axes[2].legend()

    label_font_size = 10
    axes[0].set_xlabel("Trials", fontsize=label_font_size)
    axes[0].set_ylabel("JSD", fontsize=label_font_size)

    axes[1].set_xlabel("Trials", fontsize=label_font_size)
    axes[1].set_ylabel("JSD", fontsize=label_font_size)

    axes[2].set_xlabel("Trials", fontsize=label_font_size)
    axes[2].set_ylabel("JSD", fontsize=label_font_size)

    for ax in axes:
        ax.tick_params(labelsize=7)


    figure.suptitle("Mean Convergence Plot", fontsize=14, fontweight="bold")

    plt.show()

