from sklearn.gaussian_process import GaussianProcessRegressor
import numpy as np
from scipy.stats import norm
import numpy.typing as npt
import random
import os
from tqdm import tqdm
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from gpal import plot_GP
import warnings
warnings.filterwarnings('ignore')

def outlier_detect(gpr: GaussianProcessRegressor, 
                   fit_data_X: npt.NDArray, 
                   obs_data_Y: npt.NDArray,
                   predict_candidate_X: npt.NDArray,
                   num_trials: int, 
                   init_index: int, 
                   alpha: float):

    lower_bound=1e-2
    new_kernel=gpr.kernel.clone_with_theta(gpr.kernel.theta)
    new_kernel.set_params(k2__noise_level_bounds=(lower_bound, 1e5))
    
    filtered_data_X=fit_data_X[:init_index]
    filtered_data_Y=obs_data_Y[:init_index]
    
    in_data_X=filtered_data_X  # Initialization
    in_data_Y=filtered_data_Y  # Initialization
    
    for tIdx in range(1, num_trials-1):
        new_gpr=GaussianProcessRegressor(new_kernel, normalize_y=True, n_restarts_optimizer=100)
        if tIdx<init_index:
            new_gpr.fit(fit_data_X[:tIdx], obs_data_Y[:tIdx])
            continue
        
        #expanded_data_X=np.concat((in_data_X, np.expand_dims(fit_data_X[tIdx+1], axis=-1)))
        #expanded_data_Y=np.concat((in_data_Y, np.expand_dims(obs_data_Y[tIdx+1], axis=-1)))

        if tIdx%1==0:
            new_gpr.fit(in_data_X, in_data_Y)
            post_mean_per_data, post_std_per_data=new_gpr.predict(fit_data_X[:tIdx+1], return_std=True)
            exclude_index=[]
            
            for i, (pm, pstd) in enumerate(zip(post_mean_per_data, post_std_per_data)):
                post_prob=norm.cdf(obs_data_Y[i], loc=pm, scale=pstd)
                if (post_prob<alpha or post_prob>1-alpha):
                    exclude_index.append(i)
            if len(exclude_index)>=(tIdx+1)//2:
                random.shuffle(exclude_index)
                exclude_index=exclude_index[:(tIdx+1)//2]
            include_index=np.array([i for i in range(tIdx+1) if not i in exclude_index], dtype=int)
            in_data_X=fit_data_X[include_index]
            in_data_Y=obs_data_Y[include_index]
        else:
            new_gpr.fit(in_data_X, in_data_Y)
            in_data_X=np.concat((in_data_X, np.expand_dims(fit_data_X[tIdx], -1)))
            in_data_Y=np.concat((in_data_Y, np.expand_dims(obs_data_Y[tIdx], -1)))

        #new_gpr.fit(in_data_X, in_data_Y)
        #post_mean, post_std=new_gpr.predict(predict_candidate_X, return_std=True)
        #post_mean_array[tIdx]=post_mean
        #post_std_array[tIdx]=post_std
        #if tIdx==num_trials-1:
        #    break
    final_gpr=GaussianProcessRegressor(new_kernel, normalize_y=True, n_restarts_optimizer=100)
    final_gpr.fit(in_data_X, in_data_Y)
    post_mean, post_std=final_gpr.predict(predict_candidate_X, return_std=True)
    return {'in_data_X':in_data_X, 
            'in_data_Y':in_data_Y.reshape(-1,1), 
            'post_mean':post_mean, 
            'post_std':post_std}




def outlier_detect_all(figsize,
                    opt:str, 
                    alpha_list:list[float],
                    init_index:int,
                    results_dir:str,
                    base_dir: str, 
                    column_name_X:str,
                    column_name_Y:str, 
                    predict_candidates_X):

    sbj_IDs=[int(name[-8:]) for name in os.listdir(results_dir)]
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)

    for sID in tqdm(sbj_IDs):
        sbj_dir=os.path.join(results_dir, f'participant_{sID}')
        gpr_dir=os.path.join('models', f"{sID}")
        fig_dir=os.path.join(base_dir, f"{sID}")
        if not os.path.exists(os.path.join(base_dir, f"{sID}")):
            os.mkdir(os.path.join(base_dir, f"{sID}"))

        filePath=os.path.join(sbj_dir, f"{opt}_results_{sID}.csv")
        if os.path.exists(filePath):
            df=pd.read_csv(filePath)
        else:
            raise FileNotFoundError(f"The following file cannot not found: {filePath}.")

        gprPath=os.path.join(gpr_dir, f"GPR_{sID}.pkl")
        with open(gprPath, 'rb') as f:
            gpr=pickle.load(f)

        gns=df[column_name_X].to_numpy()
        ests=df[column_name_Y].to_numpy()
        gns=np.expand_dims(gns, -1)
        fit_data_X=gns
        obs_data_Y=ests
        num_trials=fit_data_X.shape[0]


        for alpha in alpha_list:
            result=outlier_detect(gpr=gpr, 
                                fit_data_X=fit_data_X, 
                                obs_data_Y=obs_data_Y,
                                predict_candidate_X=predict_candidates_X,
                                num_trials=num_trials,
                                init_index=init_index,
                                alpha=alpha)
        
        
        
            inliers_df=pd.DataFrame(np.concat((result['in_data_X'], result['in_data_Y']), axis=1),
                                    columns=["Feature Stimulus", "Responses"])
            inliers_df.to_csv(os.path.join(base_dir, f"{sID}", f'Inliers_{opt}_{alpha}.csv'), index=False)

            figure, ax1=plot_GP(gp_regressor=gpr,
                                dataframe=inliers_df,
                                x_range=(0.0, 500.0),
                                y_range=(0.0, 600.0),
                                x_num=100,
                                column_names_specified=["Feature Stimulus", "Responses"],
                                trial_numbers_specified=None,
                                x_label_name="Stimulus Feature",
                                figure_size=(8,6))
            
            ax1.set_xlim(0, 500)
            ax1.set_ylim(0, 600)
            ax1.set_xticks(np.arange(0, 500, 50))
            ax1.tick_params(axis='x', labelsize=14)
            ax1.set_yticks(np.linspace(0, 600, 13))
            ax1.tick_params(axis='y', labelsize=14)
            
            figure.savefig(os.path.join(fig_dir, f"uncertainty_inliners_{opt}_{alpha}.png"))
