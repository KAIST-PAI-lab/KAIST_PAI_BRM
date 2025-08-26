from gpal.gpal_plot import plot_GPAL_uncertainty, plot_frequency_histogram_1D, plot_GPAL_compare_uncertainty
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
                  gpr_dir: str, 
                  sbj_dir: str, 
                  fig_dir: str,
                  sbj_id: int, 
                  predict_candidates_X:NDArray, 
                  trial_idx:int):
    
    filePath=os.path.join(sbj_dir, f"gpal_results_{sbj_id}.csv")
    if os.path.exists(filePath):
        df=pd.read_csv(filePath)
    else:
        raise FileNotFoundError(f"The following file cannot not found: {filePath}.")

    gprPath=os.path.join(gpr_dir, f"GPR_{sbj_id}.pkl")
    with open(gprPath, 'rb') as f:
        gpr=pickle.load(f)

    df_fixed=df[df['size_control']==False]
    gns=df_fixed['given_number'].to_numpy()
    ests=df_fixed['estimation'].to_numpy()
    gns=np.expand_dims(gns, -1)
    fit_data_X=gns[:trial_idx+1]
    obs_data_Y=ests[:trial_idx+1]

    new_noise_lower = 5e-3
    original_kernel=gpr.kernel
    new_noise_kernel=original_kernel.clone_with_theta(original_kernel.theta)
    new_noise_kernel.set_params(k2__noise_level_bounds=(new_noise_lower, 1e5))


    gpr=GaussianProcessRegressor(new_noise_kernel, normalize_y=True, n_restarts_optimizer=100)
    for fit_index in range(1, trial_idx+1):
        gpr.fit(fit_data_X[:fit_index], obs_data_Y[:fit_index])
    print(f"The fitted kernel: {gpr.kernel_}")
    fitted_noise=np.exp(gpr.kernel_.theta[-1])

    post_mean_final, post_stdev_final = gpr.predict(np.expand_dims(predict_candidates_X, -1), return_std=True)

    title=f"Subject #{sbj_id}, Noise Level: {fitted_noise:.4f}"
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir, exist_ok=True)
 
    fig, ax= plot_GPAL_uncertainty(fig_size=fig_size,
                                    fit_data_X=fit_data_X, 
                                    obs_data_Y=obs_data_Y, 
                                    predict_candidates_X=predict_candidates_X, 
                                    post_mean=post_mean_final, 
                                    post_stdev=post_stdev_final, 
                                    x_label='Given Number', 
                                    y_label='Estiamte', 
                                    title=title, 
                                    sigma_coef=1.0)

    filename=os.path.join(fig_dir, f"uncertainty_{sbj_id}_initialized_{new_noise_lower}.png")
    fig.savefig(filename)
    '''
    gpr2=GaussianProcessRegressor(original_kernel, normalize_y=True, n_restarts_optimizer=100)
    for fit_index in range(1, trial_idx):
        gpr2.fit(fit_data_X[:fit_index], obs_data_Y[:fit_index])
    post_mean_previous, post_stdev_previous=gpr.predict(np.expand_dims(predict_candidates_X, -1), return_std=True)
    max_stdev_design=predict_candidates_X[np.argmax(post_stdev_previous)]
    gpr2.fit(fit_data_X[:trial_idx+1], obs_data_Y[:trial_idx+1])
    post_mean_target, post_stdev_target=gpr.predict(np.expand_dims(predict_candidates_X, -1), return_std=True)
    fig, ax = plot_GPAL_compare_uncertainty(fig_size=fig_size, 
                                            font_size=16,
                                            fit_data_X=fit_data_X,
                                            obs_data_Y=obs_data_Y,
                                            predict_candidates_X=predict_candidates_X,
                                            post_mean_previous=post_mean_previous,
                                            post_stdev_previous=post_stdev_previous,
                                            post_mean_target=post_mean_target,
                                            post_stdev_target=post_stdev_target,
                                            title=f"Subject #{sbj_id}",
                                            title_previous=f"Trial {trial_idx-1}",
                                            title_target=f"Trial #{trial_idx}",
                                            max_stdev_design=max_stdev_design,
                                            sigma_coef=1.0
                                            )

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
    figsize=(16, 8)
    n_trials=20
    pred_cand_X=np.linspace(5, 500, (500-5)//5+1)
    opt='gpal'
    results_dir=os.path.join('experiment_results', '')

    sbj_IDs=[int(dirname[-8:]) for dirname in os.listdir(results_dir)]
    for sbjID in sbj_IDs:
        sbj_dir=os.path.join('experiment_results', f'participant_{sbjID}')
        gpr_dir=os.path.join('models', f"{sbjID}")
        fig_dir=os.path.join('figures', 'fixed_size', f"{sbjID}")


        plot_GPAL_fit(fig_size=figsize, 
                      gpr_dir=gpr_dir, 
                      sbj_dir=sbj_dir, 
                      fig_dir=fig_dir,
                      sbj_id=sbjID, 
                      predict_candidates_X=pred_cand_X, 
                      trial_idx=9+1)
