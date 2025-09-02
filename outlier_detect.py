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
from gpal.gpal_plot import plot_GPAL_uncertainty
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




if __name__=="__main__":

    results_dir=os.path.join('experiment_results', '')
    sbj_IDs=[int(dirname[-8:]) for dirname in os.listdir(results_dir)]
    opt='gpal'
    predict_candidate_X=np.arange(5, 501, 1).reshape(-1, 1)
    alpha=0.1
    num_trials=20
    init_index=5
    for sID in tqdm(sbj_IDs):
        sbj_dir=os.path.join('experiment_results', f'participant_{sID}')
        gpr_dir=os.path.join('models', f"{sID}")
        fig_dir=os.path.join('figures', f"{sID}")

        filePath=os.path.join(sbj_dir, f"gpal_results_{sID}.csv")
        if os.path.exists(filePath):
            df=pd.read_csv(filePath)
        else:
            raise FileNotFoundError(f"The following file cannot not found: {filePath}.")

        gprPath=os.path.join(gpr_dir, f"GPR_{sID}.pkl")
        with open(gprPath, 'rb') as f:
            gpr=pickle.load(f)

        gns=df['given_number'].to_numpy()
        ests=df['estimation'].to_numpy()
        gns=np.expand_dims(gns, -1)
        fit_data_X=gns
        obs_data_Y=ests

        result=outlier_detect(gpr=gpr, 
                              fit_data_X=fit_data_X, 
                              obs_data_Y=obs_data_Y,
                              predict_candidate_X=predict_candidate_X,
                              num_trials=num_trials,
                              init_index=init_index,
                              alpha=alpha)
        
        base_dir=os.path.join('outlier_related', f"{sID}")
        if not os.path.exists(base_dir):
            os.mkdir(base_dir)
        
        inliers_df=pd.DataFrame(np.concat((result['in_data_X'], result['in_data_Y']), axis=1),
                                columns=["Feature Stimulus", "Responses"])
        inliers_df.to_csv(os.path.join(base_dir, f'lnliers.csv'), index=False)

        figsize=(12,8)
        figure, ax=plot_GPAL_uncertainty(fig_size=figsize,
                                         fit_data_X=result['in_data_X'],
                                         obs_data_Y=result['in_data_Y'].squeeze(),
                                         predict_candidates_X=predict_candidate_X,
                                         post_mean=result['post_mean'],
                                         post_stdev=result['post_std'],
                                         x_label='Feature Stimulus',
                                         y_label='Responses',
                                         title="Uncertainty - Inlineres only")
        figure.savefig(os.path.join(base_dir, f"uncertainty_inliners.png"))
