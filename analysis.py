from gpal.utils import plotEstim1D, plotFreq1D, plotStd1D
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.typing import NDArray
from typing import Tuple, Optional

def plotGPAL(figsize: Tuple[int, int], sbjID:int, x_pred:NDArray, trial_idx:int):
    filePath=os.path.join('results', f"gpal_results_{sbjID}.csv")
    if os.path.exists(filePath):
        df=pd.read_csv(filePath)
    else:
        raise FileNotFoundError(f"The following file is not found: {filePath}")
    
    gns=df['given_number'].to_numpy()
    ests=df['estimation'].to_numpy()
    mean=df['posterior_mean'].to_numpy()
    std=df['posterior_std'].to_numpy()

    meanArrPath=os.path.join('results', f"gpal_posterior_mean_{sbjID}.csv")
    if os.path.exists(meanArrPath):
        meanArr=np.loadtxt(meanArrPath, delimiter=',')
    stdArrPath=os.path.join('results', f"gpal_posterior_std_{sbjID}.csv")
    if os.path.exists(stdArrPath):
        stdArr=np.loadtxt(stdArrPath, delimiter=',')
    
    plotStd1D(figsize, gns[:trial_idx], x_pred, ests[:trial_idx], meanArr[trial_idx], stdArr[trial_idx],
               "Given Number", "Estimate", f"Posterior mean and std: Subject #{sbjID}, Trial #{trial_idx}", sigma_coef=1.0)
    

    plotEstim1D(figsize, gns, ests, "Given Number", "Estimate", 
                f"Given Number and Estimates: Subject #{sbjID}")

def plotFreq(figsize:Tuple[int, int], n_trials: int, bin:int, ranges:Optional[Tuple[float, float]], mode:str='sum'):
    
    filenames=[]
    contents=os.listdir('results')
    for file in contents:
        target='gpal_results_'
        if target==file[:len(target)]:
            filenames.append(file)
    
    N=len(filenames)
    gns=np.zeros((N, n_trials-1))
    for i, file in enumerate(filenames):
        df=pd.read_csv(os.path.join('results', file))
        gns[i]=df['given_number'][1:]
    gns=gns.ravel()
    plotFreq1D(figsize, N, gns, bin, ranges, mode='average', xlabel="Given Number (Optimized)", title=f"Design selection frequencies ({mode})")



if __name__=="__main__":
    figsize=(10,8)
    n_trials=20
    x_pred=np.linspace(5, 500, (500-5)//5+1)
    #plotGPAL(figsize, 20190169, x_pred, trial_idx=20)
    plotFreq(figsize, n_trials, bin=10, ranges=(0.0, 500.0), mode='sum')