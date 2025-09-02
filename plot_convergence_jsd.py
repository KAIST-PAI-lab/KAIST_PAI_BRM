import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import multivariate_normal, norm, entropy
from scipy.spatial.distance import jensenshannon
from scipy.special import logsumexp
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import WhiteKernel
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def get_kl_array(x_data_array:npt.NDArray, y_data_array:npt.NDArray, num_trials:int):
    
    x_data_2D=x_data_array.reshape(-1,1)
    x_data_predict=np.arange(5, 500+5, 5).reshape(-1,1)
    kernel=C(1.0)*RBF(1.0)+WhiteKernel(0.01, (1e-2, 1e5))

    post_mean_array=np.zeros((num_trials, len(x_data_predict)))
    post_cov_array=np.zeros((num_trials, len(x_data_predict), len(x_data_predict)))
    #post_std_array=np.zeros((num_trials, len(x_data_predict)))

    gpr=GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=100)

    for idx in range(1, num_trials+1):
        x_collected=x_data_2D[:idx]
        y_collected=y_data_array[:idx]
        gpr.fit(x_collected, y_collected)
        post_mean, post_cov=gpr.predict(x_data_predict, return_cov=True)
        post_mean_array[idx-1]=post_mean
        post_cov_array[idx-1]=post_cov
    '''
    probs_array=np.zeros((num_trials, x_data_predict.shape[0]))
    x_data_predict=x_data_predict.squeeze()
    for tIdx in range(num_trials):
        probs=np.zeros_like(x_data_predict, dtype=np.float64)
        for dpIdx in range(x_data_predict.shape[0]):
            mu=post_mean_array[tIdx, dpIdx]
            sigma=post_std_array[tIdx, dpIdx]
            trial_probs=norm.pdf(x_data_predict, loc=mu, scale=sigma)
            trial_probs/=trial_probs.sum()
        probs+=trial_probs
        print(np.sum(probs))
        probs=probs/len(x_data_predict)
        probs_array[tIdx]=probs
    
    jsd_array=np.zeros((num_trials,))
    for nt in range(num_trials):
        ref_probs=probs_array[-1]
        trial_probs=probs_array[nt]
        jsd=np.square(jensenshannon(ref_probs, trial_probs, base=2))
        jsd_array[nt]=jsd
    '''
    kl_array=np.zeros((num_trials,)) 
    for tIdx in range(num_trials):
        alpha=np.eye(x_data_predict.shape[0])*1e-6
        ref_cov=post_cov_array[-1]+alpha
        trial_cov=post_cov_array[tIdx]+alpha
        ref_mean=post_mean_array[-1]
        trial_mean=post_mean_array[tIdx]

        trace_term=np.trace(np.linalg.inv(trial_cov)@ref_cov)
        squared_term=(trial_mean-ref_mean).T@np.linalg.inv(trial_cov)@(trial_mean-ref_mean)
        _, slog_trial=np.linalg.slogdet(trial_cov)
        _, slog_ref=np.linalg.slogdet(ref_cov)
        log_term=slog_trial-slog_ref
        kl=0.5*(trace_term+squared_term+log_term-x_data_predict.shape[0])
        kl_array[tIdx]=kl
        print(f"log_term : {log_term}")
        '''
        ref_dis=multivariate_normal(post_mean_array[-1], ref_cov, allow_singular=False)
        trial_dis=multivariate_normal(post_mean_array[tIdx], trial_cov, allow_singular=False)

        num_sample=2000
        ref_sample=ref_dis.rvs(size=num_sample)
        trial_sample=trial_dis.rvs(size=num_sample)

        log_ref=ref_dis.pdf(ref_sample)
        log_trial=trial_dis.logpdf(sample)
        entropy(log_ref, log_trial)

        log_m=logsumexp(np.vstack([log_ref, log_trial]), axis=0)-np.log(2.0)

        jsd=0.5*(np.mean(log_ref[:num_sample//2] - log_m[:num_sample//2])+np.mean(log_trial[num_sample//2:]-log_m[num_sample//2:]))
        jsd=jsd/np.log(2.0)
        jsd_array[tIdx-5]=jsd
        '''
    return kl_array



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

    num_kls=20

    kl_ado=np.zeros((num_subjects, num_kls))
    kl_gpal=np.zeros((num_subjects, num_kls))
    kl_bd=np.zeros((num_subjects, num_kls))
    kl=np.stack([kl_ado, kl_gpal, kl_bd], axis=0)
    
    sIndices=[i for i in range(num_subjects) if i!=7]
    for sIdx in tqdm(sIndices):
        kl[0][sIdx]=get_kl_array(given_numbers_ado[sIdx], estimates_ado[sIdx], num_trials)
        kl[1][sIdx]=get_kl_array(given_numbers_gpal[sIdx], estimates_gpal[sIdx], num_trials)
        kl[2][sIdx]=get_kl_array(given_numbers_bd[sIdx], estimates_bd[sIdx], num_trials)

    kl=kl[:,sIndices,:]
    kl_mean=np.mean(kl, axis=1)
    np.savetxt("kl_mean.csv", kl_mean, delimiter=',')

    '''
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

    '''