from gpal.utils import plotEstim1D, plotFreq1D, plotStd1D
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.typing import NDArray
from typing import Tuple

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
    
    #plotStd1D(figsize, gns[:trial_idx], x_pred, ests[:trial_idx], meanArr[trial_idx], stdArr[trial_idx],
    #           "Given Number", "Estimate", f"Posterior mean and std: Subject #{sbjID}, Trial #{trial_idx}", sigma_coef=1.0)
    
    
    plotEstim1D(figsize, gns[:trial_idx], ests[:trial_idx], "Given Number", "Estimate", 
                f"Given Number and Estimates: Subject #{sbjID}, Trial #{trial_idx}")

if __name__=="__main__":
    figsize=(10,8)
    x_pred=np.linspace(5, 500, (500-5)//5+1)
    plotGPAL(figsize, 20190169, x_pred, 20)