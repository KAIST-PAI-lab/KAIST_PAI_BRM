from gpalexp import plot_GP, plot_selection_frequency, plot_convergence
from outlier_detect import outlier_detect_all
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
from gpalexp.utils import BoundsType
from tqdm import tqdm
import pickle
import re
import warnings
warnings.filterwarnings('ignore')

def plot_GPAL_fit(gpr_dir: str, 
                  sbj_dir: str, 
                  sbj_id: int,
                  file_name:str,
):
    
    filePath=os.path.join(sbj_dir, file_name)
    if os.path.exists(filePath):
        df=pd.read_csv(filePath)
    else:
        raise FileNotFoundError(f"The following file cannot not found: {filePath}.")

    gprPath=os.path.join(gpr_dir, f"GPR_{sbj_id}.pkl")
    with open(gprPath, 'rb') as f:
        gpr=pickle.load(f)

    #gns=df[column_name_X].to_numpy()
    #ests=df[column_name_Y].to_numpy()
    #gns=np.expand_dims(gns, -1)
    #fit_data_X=gns[:trial_idx+1]
    #obs_data_Y=ests[:trial_idx+1]

    new_noise_lower = 5e-2
    original_kernel=gpr.kernel
    new_noise_kernel=original_kernel.clone_with_theta(original_kernel.theta)
    new_noise_kernel.set_params(k2__noise_level_bounds=(new_noise_lower, 1e5))


    
    gpr=GaussianProcessRegressor(new_noise_kernel, normalize_y=True, n_restarts_optimizer=100)
    #for fit_index in range(1, trial_idx+1):
    #    gpr.fit(fit_data_X[:fit_index], obs_data_Y[:fit_index])
    #fitted_noise=np.exp(gpr.kernel_.theta[-1])

    #post_mean_final, post_stdev_final = gpr.predict(predict_candidates_X, return_std=True)


 
    fig, axes = plot_GP(gp_regressor=gpr,
                      dataframe=df,
                      x_range=(0.0, 500.0),
                      y_range=(0.0, 650.0),
                      x_num=100,
                      column_names_specified=['given_number', 'estimation'],
                      trial_numbers_specified=None,
                      figure_size=(8,6),
                      sigma_coefficient=1.0,
                      x_label_name="Stimulus Feature",
                      y_label_name="Response")
    
    '''
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
    '''
    return fig, axes
'''    
def plot_GPAL_fit_compare(fig_size: Tuple[int, int],
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

    
    gpr2=GaussianProcessRegressor(original_kernel, normalize_y=True, n_restarts_optimizer=100)
    for fit_index in range(1, trial_idx+1):
        gpr2.fit(fit_data_X[:fit_index], obs_data_Y[:fit_index])
    post_mean_previous, post_stdev_previous=gpr2.predict(predict_candidates_X, return_std=True)
    gpr2.fit(fit_data_X[:trial_idx+1], obs_data_Y[:trial_idx+1])
    post_mean_target, post_stdev_target=gpr2.predict(predict_candidates_X, return_std=True)
    max_stdev_design=predict_candidates_X[np.argmax(post_stdev_target)].item()
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

def plotFreq(dir:str, opt:str, sbjID:int, bin:int, ranges:Tuple[float, float], mode:str='sum'):
    '''
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
    gns_accum=gns_accum.reshape(-1,1)
    '''
    df=pd.read_csv(os.path.join(dir, f"{opt}_results_{sbjID}.csv"))
    fig, ax = plot_selection_frequency(dataframe=df,
                                       bins=bin,
                                       val_range=ranges,
                                       column_names_specified='given_number',
                                       figure_size=(8,6),
                                       mode=mode)
    
    return fig, ax

def plotConv(gpr_dir: str, 
            sbj_dir: str, 
            sbj_id: int,
            file_name:str,
            ):
    
    filePath=os.path.join(sbj_dir, file_name)
    if os.path.exists(filePath):
        df=pd.read_csv(filePath)
    else:
        raise FileNotFoundError(f"The following file cannot not found: {filePath}.")

    gprPath=os.path.join(gpr_dir, f"GPR_{sbj_id}.pkl")
    with open(gprPath, 'rb') as f:
        gpr=pickle.load(f)
    

    fig, axes, _ = plot_convergence(gp_regressor=gpr,
                               dataframe=df,
                               x_range=(0.0, 500.0),
                               y_range=None,
                               x_num = 100,
                               column_names_specified=["given_number", "estimation"])
    return fig, axes



if __name__=="__main__":
    n_trials=20
    pred_cand_X=np.linspace(5, 520, (520-5)//5+1).reshape(-1,1)
    opts=['ado', 'random', 'gpal']
    results_dir=os.path.join('experiment_results')
    alpha=0.005

    sbj_IDs=[int(name[-8:]) for name in os.listdir(results_dir)]
    for opt in opts:
        for sbjID in tqdm(sbj_IDs):
            ref_dir=os.path.join(results_dir, f"participant_{sbjID}")
            out_dir=os.path.join('outlier_related', f'{sbjID}')
            gpr_dir=os.path.join('models', f"{sbjID}")
            fig_dir=os.path.join('figures', f"{sbjID}")
            if not os.path.exists(fig_dir):
                os.makedirs(fig_dir, exist_ok=True)
            
            fig1, axes= plot_GPAL_fit(gpr_dir=gpr_dir,
                                    sbj_dir=ref_dir,
                                    sbj_id=sbjID,
                                    file_name=f"{opt}_results_{sbjID}.csv",)
            #ax1.set_xlim(0, 500)
            #ax1.set_ylim(0, 650)
            #ax1.set_xticks(np.arange(0, 500, 50))
            #ax1.tick_params(axis='x', labelsize=14)
            #ax1.set_yticks(np.arange(0, 650, 50))
            #ax1.tick_params(axis='y', labelsize=14)

            filename1=os.path.join(fig_dir, f"uncertainty_{opt}_{sbjID}.png")
            fig1.savefig(filename1)

            
            '''
            fig2, ax1, ax2 = plot_GPAL_fit_compare(fig_size=(24,9),
                                                opt=opt,
                                                gpr_dir=gpr_dir,
                                                sbj_dir=ref_dir,
                                                fig_dir=fig_dir,
                                                sbj_id=sbjID,
                                                file_name=f"{opt}_results_{sbjID}.csv",
                                                column_name_X=f"given_number",
                                                column_name_Y=f"estimation",
                                                predict_candidates_X=pred_cand_X,
                                                title="Uncertainty Plot - Trial #6 vs Trial #7",
                                                trial_idx=6)
            ax1.set_xlim(0, 520)
            ax1.set_ylim(0, 650)
            ax1.set_xticks(np.arange(0, 520, 50))
            ax1.tick_params(axis='x', labelsize=14)
            ax1.set_yticks(np.arange(0, 550, 50))
            ax1.tick_params(axis='y', labelsize=14)
            ax2.set_xlim(0, 520)
            ax2.set_ylim(0, 650)
            ax2.set_xticks(np.arange(0, 520, 50))
            ax2.tick_params(axis='x', labelsize=14)
            ax2.set_yticks(np.arange(0, 550, 50))
            ax2.tick_params(axis='y', labelsize=14)

            filename=os.path.join(fig_dir, f"uncertainty_{sbjID}_compare_6.png")
            fig2.savefig(filename)


            fig3, ax1, ax2 = plot_GPAL_fit_compare(fig_size=(24,9),
                                                opt=opt,
                                                gpr_dir=gpr_dir,
                                                sbj_dir=ref_dir,
                                                fig_dir=fig_dir,
                                                sbj_id=sbjID,
                                                file_name=f"{opt}_results_{sbjID}.csv",
                                                column_name_X=f"given_number",
                                                column_name_Y=f"estimation",
                                                predict_candidates_X=pred_cand_X,
                                                title="Uncertainty Plot - Trial #12 vs Trial #13",
                                                trial_idx=12)
            
            ax1.set_xlim(0, 520)
            ax1.set_ylim(0, 650)
            ax1.set_xticks(np.arange(0, 520, 50))
            ax1.tick_params(axis='x', labelsize=14)
            ax1.set_yticks(np.arange(0, 550, 50))
            ax1.tick_params(axis='y', labelsize=14)
            ax2.set_xlim(0, 520)
            ax2.set_ylim(0, 650)
            ax2.set_xticks(np.arange(0, 520, 50))
            ax2.tick_params(axis='x', labelsize=14)
            ax2.set_yticks(np.arange(0, 550, 50))
            ax2.tick_params(axis='y', labelsize=14)
            
            filename=os.path.join(fig_dir, f"uncertainty_{sbjID}_compare_12.png")
            fig3.savefig(filename)


            


            '''
            if opt=='gpal':
                fig2, ax2 = plotFreq(dir=ref_dir,
                                opt=opt,
                                sbjID=sbjID,
                                bin=10,
                                ranges=(0.0, 500.0),
                                mode='average')
                
                ax2.set_xlim(0, 500)
                ax2.set_ylim(0, 1)
                ax2.set_xticks(np.arange(0, 500, 50))

                filename2=os.path.join(fig_dir, f"frequency_average_{sbjID}.png")
                fig2.savefig(filename2)

                fig3, ax3 = plotFreq(dir=ref_dir,
                                opt=opt,
                                sbjID=sbjID,
                                bin=10,
                                ranges=(0.0,500.0),
                                mode='sum')
                
                ax3.set_xlim(0, 500)
                ax3.set_xticks(np.arange(0, 500, 50))
                ax3.tick_params(axis='x', labelsize=14)

                filename3=os.path.join(fig_dir, f"frequency_sum_{sbjID}.png")
                fig3.savefig(filename3)

                
                fig4, ax4=plotConv(gpr_dir=gpr_dir,
                                sbj_dir=ref_dir,
                                sbj_id=sbjID,
                                file_name=f"{opt}_results_{sbjID}.csv")

                filename4=os.path.join(fig_dir, f"convergence_{sbjID}.png")
                fig4.savefig(filename4)
        

    
    outlier_detect_all(figsize=(10, 6),
                           opt='gpal',
                           alpha_list=[0.005],
                           init_index=5,
                           results_dir=results_dir,
                           base_dir=os.path.join(f"outlier_related"),
                           column_name_X=f"given_number",
                           column_name_Y=f"estimation",
                           predict_candidates_X=pred_cand_X
        )