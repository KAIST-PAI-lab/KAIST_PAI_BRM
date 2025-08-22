from gpal.gpal_plot import plot_GPAL_uncertainty, plot_GPAL_compare_uncertainty
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





def plot_GPAL_fit(fig_size: Tuple[int, int], gpr_dir: str, sbj_dir: str, fig_dir: str,
                   sbj_id: int, predict_candidates_X:NDArray, trial_idx:int, outlier_thres:float):
    
    filePath=os.path.join(sbj_dir, f"gpal_results_{sbj_id}.csv")
    if os.path.exists(filePath):
        df=pd.read_csv(filePath)
    else:
        raise FileNotFoundError(f"The following file cannot not found: {filePath}.")

    gprPath=os.path.join(gpr_dir, f"GPR_{sbj_id}.pkl")
    with open(gprPath, 'rb') as f:
        gpr=pickle.load(f)

    gns=df['given_number'].to_numpy()
    ests=df['estimation'].to_numpy()
    gns=np.expand_dims(gns, -1)
    ests=np.expand_dims(ests, -1)

    original_kernel=gpr.kernel
    print(f"The original kernel: {original_kernel}")

    '''
    train_index=0
    chosen_index=0
    fit_data_X=np.zeros((gns.shape[0],1))
    obs_data_Y=np.zeros((ests.shape[0],))
    gpr=GaussianProcessRegressor(kernel=original_kernel, normalize_y=True, n_restarts_optimizer=100)
    while train_index<gns.shape[0]:
        if train_index==0:
            fit_data_X[chosen_index]=gns[train_index]
            obs_data_Y[chosen_index]=ests[train_index]
            gpr.fit(fit_data_X[:chosen_index+1], obs_data_Y[chosen_index+1])
            chosen_index=chosen_index+1
            train_index=train_index+1
        else:
            post_mean, post_stdev=gpr.predict(np.expand_dims(predict_candidates_X, -1), return_std=True)
            max_stdev_index=np.argmax(post_stdev)
            post_mean_stdev=np.std(post_mean)
            if max_stdev_index < outlier_thres * post_mean_stdev:
                fit_data_X[chosen_index]=gns[train_index]
                obs_data_Y[chosen_index]=ests[train_index]
                chosen_index=chosen_index+1
            train_index=train_index+1
            gpr.fit(fit_data_X[:chosen_index+1], obs_data_Y[:chosen_index+1])
    fit_data_X=fit_data_X[:chosen_index]
    obs_data_Y=obs_data_Y[:chosen_index]
    '''
    
    gpr=GaussianProcessRegressor(original_kernel, normalize_y=True, n_restarts_optimizer=100)
    for fit_index in range(1, chosen_index+1):
        gpr.fit(fit_data_X[:fit_index], obs_data_Y[:fit_index])
    print(f"The fitted kernel: {gpr.kernel_}")

    post_mean_final, post_stdev_final = gpr.predict(np.expand_dims(predict_candidates_X, -1), return_std=True)

    title=f"Subject #{sbj_id}, Trial #{trial_idx-1}"
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)
    filename=os.path.join(fig_dir, f"{sbj_id}_uncertainty_trial_{trial_idx}.png")
 
    plot_GPAL_uncertainty(fig_size=fig_size,
                            fit_data_X=fit_data_X, 
                            predict_candidates_X=predict_candidates_X, 
                            obs_data_Y=obs_data_Y, 
                            post_mean=post_mean_final, 
                            post_stdev=post_stdev_final, 
                            xlabel='Given Number', 
                            ylabel='Estiamte', 
                            title=title, 
                            file_name=filename,
                            sigma_coef=1.0)

    


def plotFreq(figsize:Tuple[int, int], dir:str, opt:str, n_trials: int, bin:int, ranges:Optional[Tuple[float, float]], mode:str='sum'):
    
    filenames=[]
    contents=os.listdir(dir)
    for file in contents:
        target=f'{opt}_results_'
        if target==file[:len(target)]:
            filenames.append(file)
    
    N=len(filenames)
    gns=np.zeros((N, n_trials-1))
    for i, file in enumerate(filenames):
        df=pd.read_csv(os.path.join(dir, file))
        gns[i]=df['given_number'][1:]
    gns=gns.ravel()
    plotFreq1D(figsize, N, gns, bin, ranges, mode='average', xlabel="Given Number (Optimized)", title=f"Design selection frequencies ({mode})")

'''
plot_GPAL_compare_uncertainty(fig_size=fig_size, 
                            font_size=, gns[:trial_idx], x_pred, ests[:trial_idx], means[trial_idx-1], stds[trial_idx-1], post_mean, post_std,
#           "Given Number", "Estimate", title=title, titleLeft=titleLeft, titleRight=titleRight, sigma_coef=1.0, maxStdDesign=maxStdDesign)
'''

if __name__=="__main__":
    figsize=(16, 8)
    n_trials=20
    pred_cand_X=np.linspace(5, 500, (500-5)//5+1)
    sbjID=25081110
    opt='gpal'
    sbj_dir=os.path.join('experiment_results', f'participant_{sbjID}')
    gpr_dir=os.path.join('models', f"{sbjID}")
    fig_dir=os.path.join('figures', f"{sbjID}")

    plot_GPAL_fit(fig_size=figsize, gpr_dir=gpr_dir, sbj_dir=sbj_dir, 
                  fig_dir=fig_dir,sbj_id=sbjID, predict_candidates_X=pred_cand_X, 
                  trial_idx=19+1, outlier_thres=2.5)
    #plotFreq(figsize, opt, dir, n_trials, bin=10, ranges=(0.0, 500.0), mode='sum')#