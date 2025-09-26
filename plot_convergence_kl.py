import math
from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import multivariate_normal, norm, entropy, t
from scipy.spatial.distance import jensenshannon
from scipy.special import logsumexp
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import WhiteKernel
from outlier_detect import outlier_detect
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def get_kl_array(x_data_array:npt.NDArray, y_data_array:npt.NDArray, num_trials:int):
    
    x_data_2D=x_data_array.reshape(-1,1)
    x_data_predict=np.arange(5, 500+1, 1).reshape(-1,1)
    kernel=C(1.0)*RBF(1.0)+WhiteKernel(0.01, (1e-2, 1e5))

    post_mean_array=np.zeros((num_trials, len(x_data_predict)))
    post_cov_array=np.zeros((num_trials, len(x_data_predict), len(x_data_predict)))


    gpr=GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=50)

    for idx in range(1, num_trials+1):
        x_collected=x_data_2D[:idx]
        y_collected=y_data_array[:idx]
        gpr.fit(x_collected, y_collected)
        post_mean, post_cov=gpr.predict(x_data_predict, return_cov=True)
        post_mean_array[idx-1]=post_mean
        post_cov_array[idx-1]=post_cov

    kl_array=np.zeros((num_trials,)) 
    for tIdx in range(num_trials):
        alpha=np.eye(x_data_predict.shape[0])*1e-10
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

    return kl_array



if __name__=="__main__":

    num_subjects=18
    num_trials=20
    alpha=0.005
    
    base_dir="experiment_results"
    '''
    base_dir="outlier_related"
    
    '''
    given_numbers_gpal = np.zeros((num_subjects, num_trials))
    given_numbers_bd = np.zeros((num_subjects, num_trials))
    estimates_gpal = np.zeros((num_subjects, num_trials))
    estimates_bd = np.zeros((num_subjects, num_trials))
    '''

    given_numbers_gpal=[]
    given_numbers_bd=[]
    estimates_gpal=[]
    estimates_bd=[]
    '''

    # Iterate the files in the data folder
    data_dir = Path(base_dir)
    subject_count = 0
    for sIdx, folder in enumerate(data_dir.iterdir()):
        subject_count += 1
        if folder.is_dir():
            print(f"subject count: {subject_count}")
            print(f"Current Folder: {folder.name}")
            participant_code = folder.name[-8:]
            print(f"Subject #{participant_code}")
            
            file_name_gpal=f"gpal_results_{participant_code}.csv"
            file_path_gpal=os.path.join(base_dir, folder.name, file_name_gpal)
            file_name_bd=f"random_results_{participant_code}.csv"
            file_path_bd=os.path.join(base_dir, folder.name, file_name_bd)
            '''            
            file_name_gpal=f"Inliers_gpal_{alpha}.csv"
            file_path_gpal=os.path.join(base_dir, participant_code, file_name_gpal)
            file_name_bd=f"Inliers_random_{alpha}.csv"
            file_path_bd=os.path.join(base_dir, participant_code, file_name_bd)
            '''
            print(file_path_gpal)
            if os.path.exists(file_path_gpal):
                print(f"Current file: {file_name_gpal}")
                df = pd.read_csv(file_path_gpal)
                
                given_numbers = df["given_number"].to_numpy()
                estimates = df["estimation"].to_numpy()
                given_numbers_gpal[sIdx]=given_numbers
                estimates_gpal[sIdx]=estimates
                
                '''
                given_numbers=df["Feature Stimulus"].to_numpy()
                estimates=df["Responses"].to_numpy()
                given_numbers_gpal.append(given_numbers)
                estimates_gpal.append(estimates)
                '''
            else:
                raise FileNotFoundError(f"File Not Found: {file_path_gpal}")
                    
            if os.path.exists(file_path_bd):
                print(f"Current file: {file_name_bd}")
                df = pd.read_csv(file_path_bd)
                
                given_numbers = df["given_number"].to_numpy()
                estimates = df["estimation"].to_numpy()
                given_numbers_bd[sIdx]=given_numbers
                estimates_bd[sIdx]=estimates
                '''
                given_numbers=df["Feature Stimulus"].to_numpy()
                estimates=df["Responses"].to_numpy()
                given_numbers_bd.append(given_numbers)
                estimates_bd.append(estimates)
                '''
            else:
                raise FileNotFoundError(f"File Not Found: {file_path_bd}")

    num_kls=20
    kl_gpal=np.zeros((num_subjects, num_kls))
    kl_bd=np.zeros((num_subjects, num_kls))
    
    sIndices=[i for i in range(num_subjects)]
    for sIdx in tqdm(sIndices):
        kl_gpal_array=get_kl_array(given_numbers_gpal[sIdx], estimates_gpal[sIdx], len(estimates_gpal[sIdx]))
        kl_gpal_array=np.concat([kl_gpal_array, np.zeros((num_kls-len(kl_gpal_array),), dtype=np.float64)])
        kl_gpal[sIdx]=kl_gpal_array
        kl_bd_array=get_kl_array(given_numbers_bd[sIdx], estimates_bd[sIdx], len(estimates_bd[sIdx]))
        kl_bd_array=np.concat([kl_bd_array, np.zeros((num_kls-len(kl_bd_array),), dtype=np.float64)])
        kl_bd[sIdx]=kl_bd_array
    '''
    kl_gpal_std=np.std(kl_gpal, axis=0)
    kl_bd_std=np.std(kl_bd, axis=0)
    '''
    kl_gpal_mean=np.zeros((num_kls,), dtype=np.float64)
    kl_bd_mean=np.zeros((num_kls, ), dtype=np.float64)
    kl_gpal_std=np.zeros((num_kls, ), dtype=np.float64)
    kl_bd_std=np.zeros((num_kls, ), dtype=np.float64)
    kl_gpal_num=np.zeros((num_kls, ), dtype=np.int64)
    kl_bd_num=np.zeros((num_kls, ), dtype=np.int64)
    #kl_gpal_sum=np.sum(kl_gpal, axis=0)
    #kl_bd_sum=np.sum(kl_bd, axis=0)
    #_, kl_gpal_nonzero=np.nonzero(kl_gpal)
    #_, kl_bd_nonzero=np.nonzero(kl_bd)
    #counts_gpal=np.bincount(kl_gpal_nonzero)
    #counts_bd=np.bincount(kl_bd_nonzero)
    #kl_gpal_mean=np.zeros((num_kls,), dtype=np.float64)
    #kl_bd_mean=np.zeros((num_kls, ), dtype=np.float64)
    for i in range(num_kls):
        indices_gpal=np.nonzero(kl_gpal[:,i])[0]
        if len(indices_gpal)>0:
            kl_gpal_mean[i]=np.mean(kl_gpal[indices_gpal,i])
        else:
            kl_gpal_mean[i]=np.nan
        if len(indices_gpal)>1:
            kl_gpal_std[i]=np.std(kl_gpal[indices_gpal,i])
        else:
            kl_gpal_std[i]=np.nan
        kl_gpal_num[i]=len(indices_gpal)
        
        indices_bd=np.nonzero(kl_bd[:,i])[0]
        if len(indices_bd)>0:
            kl_bd_mean[i]=np.mean(kl_bd[indices_bd,i])
        else:
            kl_bd_mean[i]=np.nan
        if len(indices_bd)>1:
            kl_bd_std[i]=np.std(kl_bd[indices_bd,i])
        else:
            kl_bd_std[i]=np.nan
        kl_bd_num[i]=len(indices_bd)

    
    kl_mean=np.vstack([kl_gpal_mean, kl_bd_mean]).T    
    kl_mean[-1]=0
    df=pd.DataFrame(kl_mean,
                    index=[f"Trial {i+1}" for i in range(num_trials)],
                    columns=["GPAL", "BD"])
    '''
    df.to_csv(f"kl_mean_inliers_{alpha}.csv", index=False)
    '''
    df.to_csv(f"kl_mean_original.csv", index=False)
    
    gpal_crit=np.zeros((num_kls, ), dtype=np.float64)
    bd_crit=np.zeros((num_kls, ), dtype=np.float64)
    for i in range(len(kl_gpal_num)):
        if kl_gpal_num[i]>1:
            gpal_crit[i]=t.ppf(0.975, kl_gpal_num[i]-1)
    for j in range(len(kl_bd_num)):
        if kl_bd_num[j]>1:
            bd_crit[j]=t.ppf(0.975, kl_bd_num[j]-1)

    #kl_gpal_num[kl_gpal_num==0]=np.nan
    #kl_bd_num[kl_bd_num==0]=np.nan
    
    gpal_error=gpal_crit*kl_gpal_std/(np.sqrt(kl_gpal_num))
    bd_error=bd_crit*kl_bd_std/(np.sqrt(kl_bd_num))
    gpal_error[np.isnan(gpal_error)]=0
    bd_error[np.isnan(bd_error)]=0

    trials = np.arange(1, 21)
    figure=plt.figure(figsize=(16,10))
    #figure, axes = plt.subplots(2, 1, figsize=(6, 9))
    #figure.subplots_adjust(hspace=0.5)
    beta=1.0
    
    plt.plot(
        trials[2:],
        kl_mean[2:,0],
        color="blue",
        marker="o",
        linewidth=2,
        markersize=4,
        label="GPAL",
    )

    plt.fill_between(trials[2:], kl_mean[2:,0]+beta*gpal_error[2:], kl_mean[2:,0]-beta*gpal_error[2:], alpha=0.25, color="blue")
    # --- BD plot ---
    plt.plot(trials[2:], kl_mean[2:,1], color="red", marker="o", linewidth=2, markersize=4, 
                 label="Balanced Random")
    plt.fill_between(trials[2:], kl_mean[2:,1]+beta*bd_error[2:], kl_mean[2:,1]-beta*bd_error[2:], alpha=0.25, color="red")
    
    '''
    plt.plot(trials, np.log10(kl_mean[:,0]), color="blue", marker="o", linewidth=2, markersize=4, label="GPAL")
    plt.fill_between(trials, np.log10(kl_mean[:,0]+beta*gpal_error), np.log10(kl_mean[:,0]-beta*gpal_error), alpha=0.25, color="blue")
    plt.plot(trials, np.log10(kl_mean[:,1]), color="red", marker="o", linewidth=2, markersize=4, label="Balanced Deisgn")
    plt.fill_between(trials, np.log10(kl_mean[:,1]+beta*bd_error), np.log10(kl_mean[:,1]-beta*bd_error), alpha=0.25, color="red")
    '''

    plt.ylim(bottom=0)
    plt.xticks([i for i in range(3, 21, 1)])
    label_font_size = 24
    plt.xlabel("Trials", fontsize=label_font_size)
    plt.ylabel("KL Divergence", fontsize=label_font_size)
    plt.tick_params('x', labelsize=20)
    plt.tick_params('y', labelsize=20)
    #plt.title("Mean Convergence Plot\n(Original Data)", fontsize=20, fontweight="bold")
    plt.legend(fontsize=label_font_size)
    
    figname=f"convergence_KL_alpha_{alpha}.png"
    plt.savefig(os.path.join("figures", figname))

    