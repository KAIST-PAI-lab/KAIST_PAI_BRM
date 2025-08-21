from gpal.utils import plotEstim1D, plotFreq1D, plotStd1D, plotStd1DCompare
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

def plotStdFitGPAL(figsize: Tuple[int, int], constant_value:float, cv_range:BoundsType,
                   length_scale:float, ls_range:BoundsType, noise_level:float, nl_range:BoundsType,
                   dir:str, sbjID:int, x_pred:NDArray, trial_idx:int):
    filePath=os.path.join(dir, f"{opt}_results_{sbjID}.csv")
    if os.path.exists(filePath):
        df=pd.read_csv(filePath)
    else:
        raise FileNotFoundError(f"The following file cannot not found: {filePath}.")

    gns=df['given_number'].to_numpy()
    ests=df['estimation'].to_numpy()

    kernel=ConstantKernel(constant_value, cv_range)*RBF(length_scale, ls_range)+WhiteKernel(noise_level, nl_range)
    gpr=GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=100)

    for ti in range(trial_idx):
        gpr.fit(np.expand_dims(gns[:ti+1], -1), ests[:ti+1])
        
    print(f"get_params: {gpr.kernel_.get_params()}")
    post_mean, post_std = gpr.predict(np.expand_dims(x_pred, -1), return_std=True)
    maxStdDesign=float(5*np.argmax(post_std)+5)
    cv=gpr.kernel_.k1.k1.constant_value
    ls=gpr.kernel_.k1.k2.length_scale
    nl=gpr.kernel_.k2.noise_level
    #title=f"Subject #{sbjID}, Trial #{trial_idx-1}\n constant_value = {cv:.4f},  length_scale = {ls:.4f},  noise_level = {nl:.4f}"
    title=""
    titleLeft=f"Trial #{trial_idx-1}"
    titleRight=f"Trial #{trial_idx}"
    fontsize=24
    plotStd1DCompare(figsize, fontsize, gns[:trial_idx], x_pred, ests[:trial_idx], means[trial_idx-1], stds[trial_idx-1], post_mean, post_std,
               "Given Number", "Estimate", title=title, titleLeft=titleLeft, titleRight=titleRight, sigma_coef=1.0, maxStdDesign=maxStdDesign)

    #plotStd1D(figsize, gns[:trial_idx], x_pred, ests[:trial_idx], post_mean, post_std, 'Given Number', 'Estiamte', title, 1.0)
    #plt.subplot(1,2,2)
    #plotEstim1D(figsize, gns, ests, "Given Number", "Estimate", 
    #            f"Given Number and Estimates: Subject #{sbjID}")

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


        
if __name__=="__main__":
    figsize=(18, 6)
    n_trials=20
    x_pred=np.linspace(5, 500, (500-5)//5+1)
    sbjID=25081110
    opt='gpal'
    dir=os.path.join('experiment_results', f'participant_{sbjID}')
    #dir='results'

    constant_value=1.0
    cv_range=(1e-5, 1e5)
    length_scale=1.0
    ls_range=(1e-5, 1e5)
    noise_level=0.05
    nl_range=(1e-5, 1e1)
    plotStdFitGPAL(figsize, constant_value, cv_range, length_scale, ls_range, noise_level, nl_range, 
                   dir, sbjID, x_pred, trial_idx=6+1)
    #plotFreq(figsize, opt, dir, n_trials, bin=10, ranges=(0.0, 500.0), mode='sum')#