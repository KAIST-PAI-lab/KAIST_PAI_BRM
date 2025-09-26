from gpal.gpal_plot import plot_GPAL_uncertainty, plot_frequency_histogram_1D, plot_GPAL_compare_uncertainty
from outlier_detect import outlier_detect
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from mpl_toolkits.mplot3d import Axes3D
from numpy.typing import NDArray
from scipy.interpolate import interp1d
from typing import Tuple, Optional
from gpal.utils import BoundsType
import pickle
import re
import warnings
warnings.filterwarnings('ignore')


def plot_GPAL_fit(fig_size: Tuple[int, int],
                  opt:str, 
                  gpr_dir: str, 
                  sbj_dir: str, 
                  fig_dir: str,
                  sbj_id: int,
                  file_name:str,
                  column_name_X:str,
                  column_name_Y:str, 
                  predict_candidates_X:NDArray,
                  title:str,
                  trial_idx:int):
    
    filePath=os.path.join(sbj_dir, file_name)
    if os.path.exists(filePath):
        df=pd.read_csv(filePath)
    else:
        raise FileNotFoundError(f"The following file cannot not found: {filePath}.")

    gprPath=os.path.join(gpr_dir, f"GPR_{sbj_id}.pkl")
    with open(gprPath, 'rb') as f:
        gpr=pickle.load(f)

    gns=df[column_name_X].to_numpy()
    ests=df[column_name_Y].to_numpy()
    gns=np.expand_dims(gns, -1)
    fit_data_X=gns[:trial_idx+1]
    obs_data_Y=ests[:trial_idx+1]

    new_noise_lower = 5e-2
    original_kernel=gpr.kernel
    new_noise_kernel=original_kernel.clone_with_theta(original_kernel.theta)
    new_noise_kernel.set_params(k2__noise_level_bounds=(new_noise_lower, 1e5))

    
    gpr=GaussianProcessRegressor(new_noise_kernel, normalize_y=True, n_restarts_optimizer=100)
    for fit_index in range(1, trial_idx+1):
        gpr.fit(fit_data_X[:fit_index], obs_data_Y[:fit_index])
    print(f"The fitted kernel: {gpr.kernel_}")
    fitted_noise=np.exp(gpr.kernel_.theta[-1])

    post_mean_final, post_stdev_final = gpr.predict(np.expand_dims(predict_candidates_X, -1), return_std=True)

    #title=f"Subject #{sbj_id}, Noise Level: {fitted_noise:.4f}"
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir, exist_ok=True)
 
    fig, ax= plot_GPAL_uncertainty(fig_size=fig_size,
                                    fit_data_X=fit_data_X, 
                                    obs_data_Y=obs_data_Y, 
                                    predict_candidates_X=predict_candidates_X, 
                                    post_mean=post_mean_final, 
                                    post_stdev=post_stdev_final, 
                                    x_label='Stimulus Feature', 
                                    y_label='Response', 
                                    title=title, 
                                    sigma_coef=1.0)

    #filename=os.path.join(fig_dir, f"uncertainty_{sbj_id}_initialized_{new_noise_lower}.png")
    #fig.savefig(filename)
    return fig, ax
    
    '''
    gpr2=GaussianProcessRegressor(original_kernel, normalize_y=True, n_restarts_optimizer=100)
    for fit_index in range(1, trial_idx+1):
        gpr2.fit(fit_data_X[:fit_index], obs_data_Y[:fit_index])
    post_mean_previous, post_stdev_previous=gpr2.predict(np.expand_dims(predict_candidates_X, -1), return_std=True)
    gpr2.fit(fit_data_X[:trial_idx+1], obs_data_Y[:trial_idx+1])
    post_mean_target, post_stdev_target=gpr2.predict(np.expand_dims(predict_candidates_X, -1), return_std=True)
    max_stdev_design=predict_candidates_X[np.argmax(post_stdev_target)]
    fig, ax1, ax2 = plot_GPAL_compare_uncertainty(fig_size=fig_size, 
                                            font_size=24,
                                            fit_data_X=fit_data_X,
                                            obs_data_Y=obs_data_Y,
                                            predict_candidates_X=predict_candidates_X,
                                            post_mean_previous=post_mean_previous,
                                            post_stdev_previous=post_stdev_previous,
                                            post_mean_target=post_mean_target,
                                            post_stdev_target=post_stdev_target,
                                            xlabel="Stimulus Feature",
                                            ylabel="Response",
                                            title=f"",
                                            title_previous=f"Trial #{trial_idx}",
                                            title_target=f"Trial #{trial_idx+1}",
                                            max_stdev_design=max_stdev_design,
                                            sigma_coef=1.0
                                            )
    return fig, ax1, ax2
    '''

def plotFreq(figsize:Tuple[int, int], dir:str, opt:str, n_trials: int, bin:int, ranges:Optional[Tuple[float, float]], mode:str='sum'):
    
    filenames=[]
    subject_dirs=os.listdir(dir)
    for sbj_dir in subject_dirs:
        target=f'{opt}_results_'
        sbj_files=os.listdir(os.path.join(dir, sbj_dir))
        for file in sbj_files:
            if target==file[:len(target)]:
                filenames.append(os.path.join(dir, sbj_dir, file))
    
    N=len(filenames)
    gns_accum=np.zeros((N, n_trials-1))
    for i, file in enumerate(filenames):
        df=pd.read_csv(file)
        gns_accum[i]=df['given_number'][1:]
    gns_accum=gns_accum.ravel()
    plot_frequency_histogram_1D(fig_size=figsize, 
                                num_data=N, 
                                design_var=gns_accum, 
                                bins=bin, 
                                ranges=ranges,  
                                x_label="Given Number (Optimized)",
                                y_label="Frequency",
                                title=f"Design selection frequencies ({mode})",                                
                                mode='average')



if __name__=="__main__":
    figsize=(10,6)
    n_trials=20
    pred_cand_X=np.linspace(5, 520, (520-5)//5+1)
    opt='gpal'
    results_dir=os.path.join('experiment_results', '')

    sbj_IDs=[25081314]
    for sbjID in sbj_IDs:
        ref_dir=os.path.join('experiment_results', f"participant_{sbjID}")
        out_dir=os.path.join('outlier_related', f'{sbjID}')
        gpr_dir=os.path.join('models', f"{sbjID}")
        fig_dir=os.path.join('figures', f"{opt}", f"{sbjID}")
        
        fig1, ax1 = plot_GPAL_fit(fig_size=figsize,
                                    opt=opt,
                                    gpr_dir=gpr_dir,
                                    sbj_dir=ref_dir,
                                    fig_dir=fig_dir,
                                    sbj_id=sbjID,
                                    file_name=f"{opt}_results_{sbjID}.csv",
                                    column_name_X=f"given_number",
                                    column_name_Y=f"estimation",
                                    predict_candidates_X=pred_cand_X,
                                    title="Raw Data",
                                    trial_idx=20)
        '''
        fig2, ax2 = plot_GPAL_fit(fig_size=figsize,
                                  opt=opt,
                                  gpr_dir=gpr_dir,
                                  sbj_dir=out_dir,
                                  fig_dir=fig_dir,
                                  sbj_id=sbjID,
                                  file_name=f"Inliers_{opt}_0.005.csv",
                                  column_name_X="Feature Stimulus",
                                  column_name_Y="Responses",
                                  predict_candidates_X=pred_cand_X,
                                  title="Excluding Outliers",
                                  trial_idx=14)
        '''
        
        ax1.set_xlim(0, 520)
        ax1.set_ylim(0, 550)
        ax1.set_xticks(np.arange(0, 520, 50))
        ax1.tick_params(axis='x', labelsize=14)
        ax1.set_yticks(np.arange(0, 550, 50))
        ax1.tick_params(axis='y', labelsize=14)
        '''
        ax2.set_xlim(0, 520)
        ax2.set_ylim(0, 550)
        ax2.set_xticks(np.arange(0, 520, 50))
        ax2.tick_params(axis='x', labelsize=14)
        ax2.set_yticks(np.arange(0, 550, 50))
        ax2.tick_params(axis='y', labelsize=14)

        #fig.tight_layout()
        '''
        plt.show()