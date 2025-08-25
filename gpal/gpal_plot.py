import numpy as np
import numpy.typing as npt
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D



def plot_GPAL_uncertainty(fig_size:Tuple[int, int], fit_data_X:npt.NDArray, predict_candidates_X:npt.NDArray, 
                        obs_data_Y:npt.NDArray, post_mean:npt.NDArray, post_stdev:npt.NDArray, 
                        x_label:str, y_label:str, title:str, sigma_coef:float=1.0):
    if not isinstance(fig_size, Tuple):
        raise TypeError(f"fig_size should be a tuple, got the type of {type(fig_size).__name__}.")
    if any([not isinstance(fs, int) for fs in fig_size]):
        raise ValueError(f"fig_size should contain integer elements.")
    if len(fig_size)!=2:
        raise ValueError(f"figsize should be of length 2, got {len(fig_size)}.")
    if not isinstance(fit_data_X, np.ndarray):
        raise TypeError(f"fit_data_X should be a numpy array, got the type of {type(fit_data_X).__name__}.")
    if fit_data_X.ndim!=2:
        raise ValueError(f"fit_data_X should be a 2D array, got {fit_data_X.ndim} dimensions.")
    if not isinstance(predict_candidates_X, np.ndarray):
        raise TypeError(f"predict_candidates_X should be a numpy array, got the type of {type(predict_candidates_X).__name__}.")
    if predict_candidates_X.ndim!=1:
        raise ValueError(f"predict_candidates_X should be a 1D array, got {predict_candidates_X.ndim} dimensions.")
    if not isinstance(obs_data_Y, np.ndarray):
        raise TypeError(f"obs_data_Y should be a numpy array, got the type of {type(obs_data_Y).__name__}.")
    if obs_data_Y.ndim!=1:
        raise ValueError(f"obs_data_Y should be a 1D array, got {obs_data_Y.ndim} dimensions.")
    if not isinstance(post_mean, np.ndarray):
        raise TypeError(f"post_mean should be a numpy array, got the type of {type(post_mean).__name__}.")
    if post_mean.ndim!=1:
        raise ValueError(f"post_mean should be a 1D array, got {post_mean.ndim} dimensions.")
    if not isinstance(post_stdev, np.ndarray):
        raise TypeError(f"post_stdev should be a numpy array, got {type(post_stdev).__name__}.")
    if post_stdev.ndim!=1:
        raise ValueError(f"post_stdev should be a 1D array, got {post_stdev.ndim} dimensions.")
    if fit_data_X.shape[0]!=obs_data_Y.shape[0]:
        raise ValueError(f"fit_data_X and obs_data_Y should have equal number of data, got {fit_data_X.shape[0]} and {obs_data_Y.shape[0]}.")
    if predict_candidates_X.shape[0]!=post_mean.shape[0]:
        raise ValueError(f"predict_candidates_X and post_mean should have equal number of data, got {predict_candidates_X.shape[0]} and {post_mean.shape[0]}.")
    if predict_candidates_X.shape[0]!=post_stdev.shape[0]:
        raise ValueError(f"predict_candidates_X and post_stdev should have equal number of data, got {predict_candidates_X.shape[0]} and {post_stdev.shape[0]}.")
    if not isinstance(sigma_coef, float):
        raise TypeError(f"sigma_coef should be a float value, got the type of {type(sigma_coef).__name__}.")
    if sigma_coef<0:
        raise ValueError(f"sigma_coef should be non-negative, got {sigma_coef}.")
    if not isinstance(x_label, str):
        raise TypeError(f"xlabel should be a string value, got the type of {type(x_label).__name__}.")
    if not isinstance(y_label, str):
        raise TypeError(f"ylabel should be a string value, got the type of {type(y_label).__name__}.")
    if not isinstance(title, str):
        raise TypeError(f"title should be a string value, got the type of {type(title).__name__}.")
    
    figure=plt.figure(figsize=fig_size)
    ax=figure.add_subplot(1,1,1)
    ax.scatter(fit_data_X.ravel(), obs_data_Y, c='black', label='Data')
    ax.plot(predict_candidates_X, post_mean, label="Prediction", linewidth=2.5, color='black')
    ax.fill_between(predict_candidates_X, post_mean-sigma_coef*post_stdev, 
                     post_mean+sigma_coef*post_stdev, alpha=0.3, label='Uncertainty')
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    return figure, ax





def plot_GPAL_compare_uncertainty(fig_size:Tuple[int, int], font_size:int, fit_data_X:npt.NDArray, 
                                    predict_candidates_X:npt.NDArray, obs_data_Y:npt.NDArray, 
                                    post_mean:npt.NDArray, post_stdev:npt.NDArray, 
                                    post_mean_after:npt.NDArray, post_stdev_after:npt.NDArray,
                                    title:str, title_left:str, title_right:str, 
                                    max_stdev_design:float, sigma_coef:float=1.0):
    if not isinstance(fig_size, tuple):
        raise TypeError(f"fig_size should be a tuple, got the type of {type(fig_size).__name__}")
    if any([not isinstance(fs, int) for fs in fig_size]):
        raise ValueError(f"fig_size should contain integer elements.")
    if len(fig_size)!=2:
        raise ValueError(f"fig_size should be of length 2, got {len(fig_size)}.")
    if not isinstance(font_size, int):
        raise TypeError(f"font_size should be an integer value, got the type of {type(font_size).__name__}.")
    if font_size<=0:
        raise ValueError(f"font_size should be a positive value, got {font_size}.")
    if not isinstance(fit_data_X, np.ndarray):
        raise TypeError(f"fit_data_X should be a numpy array, got the type of {type(fit_data_X).__name__}.")
    if fit_data_X.ndim!=1:
        raise ValueError(f"fit_data_X should be a 1D array, got {fit_data_X.ndim} dimensions.")
    if not isinstance(predict_candidates_X, np.ndarray):
        raise TypeError(f"predict_candidates_X should be a numpy array, got the type of {type(predict_candidates_X).__name__}.")
    if predict_candidates_X.ndim!=1:
        raise ValueError(f"predict_candidates_X should be a 1D array, got {predict_candidates_X.ndim} dimensions.")
    if not isinstance(obs_data_Y, np.ndarray):
        raise TypeError(f"obs_data_Y should be a numpy array, got the type of {type(obs_data_Y).__name__}.")
    if obs_data_Y.ndim!=1:
        raise ValueError(f"obs_data_Y should be a 1D array, got {obs_data_Y.ndim} dimensions.")
    if not isinstance(post_mean, np.ndarray):
        raise TypeError(f"post_mean should be a numpy array, got the type of {type(post_mean).__name__}.")
    if post_mean.ndim!=1:
        raise ValueError(f"post_mean should be a 1D array, got {post_mean.ndim} dimensions.")
    if not isinstance(post_stdev, np.ndarray):
        raise TypeError(f"post_stdev should be a numpy array, got the type of {type(post_stdev).__name__}.")
    if post_stdev.ndim!=1:
        raise ValueError(f"post_stdev should be a 1D array, got {post_stdev.ndim} dimensions.")
    if not isinstance(post_mean_after, np.ndarray):
        raise TypeError(f"post_mean_after should be a numpy array, got the type of {type(post_mean_after).__name__}.")
    if post_mean_after.ndim!=1:
        raise ValueError(f"post_mean_after should be a 1D array, got {post_mean_after.ndim} dimensions.")
    if not isinstance(post_stdev_after, np.ndarray):
        raise TypeError(f"post_stdev_after should be a numpy array, got the type of {type(post_stdev_after).__name__}.")
    if post_stdev_after.ndim!=1:
        raise ValueError(f"post_stdev_after should be a 1D array, got {post_stdev_after.ndim} dimensions.")
    if fit_data_X.shape[0]!=obs_data_Y.shape[0]:
        raise ValueError(f"fit_data_X and obs_data_Y should have equal number of data, got {fit_data_X.shape[0]} and {obs_data_Y.shape[0]}.")
    if predict_candidates_X.shape[0]!=post_mean.shape[0]:
        raise ValueError(f"predict_candidates_X and post_mean should have equal number of data, got {predict_candidates_X.shape[0]} and {post_mean.shape[0]}.")
    if predict_candidates_X.shape[0]!=post_stdev.shape[0]:
        raise ValueError(f"predict_candidates_X and post_stdev should have equal number of data, got {predict_candidates_X.shape[0]} and {post_stdev.shape[0]}.")
    if predict_candidates_X.shape[0]!=post_mean_after.shape[0]:
        raise ValueError(f"predict_candidates_X and post_mean_after should have equal number of data, got {predict_candidates_X.shape[0]} and {post_mean_after.shape[0]}.")
    if predict_candidates_X.shape[0]!=post_stdev_after.shape[0]:
        raise ValueError(f"predict_candidates_X and post_stdev_after should have equal number of data, got {predict_candidates_X.shape[0]} and {post_stdev_after.shape[0]}.")
    if not isinstance(max_stdev_design, float):
        raise TypeError(f"max_stdev_design should be a float value, got the type of {type(max_stdev_design).__name__}.")
    if not isinstance(sigma_coef, float):
        raise TypeError(f"sigma_coef should be a float value, got the type of {type(sigma_coef).__name__}")
    if sigma_coef<0:
        raise ValueError(f"sigma_coef should be non-negative, got {sigma_coef}.")
    if not isinstance(title, str):
        raise TypeError(f"title should be a string value, got the type of {type(title).__name__}.")
    if not isinstance(title_left, str):
        raise TypeError(f"titleLeft should be a string value, got the type of {type(title_left).__name__}.")
    if not isinstance(title_right, str):
        raise TypeError(f"titleRight should be a string value, got the type of {type(title_right).__name__}.")
    
    fig, (ax1, ax2)=plt.subplots(1,2, figsize=fig_size)
    
    ax1.scatter(fit_data_X[:-1], obs_data_Y[:-1], c='black', label='Data')
    ax1.plot(predict_candidates_X, post_mean, label="Prediction", linewidth=2.5, color='black')
    ax1.fill_between(predict_candidates_X.ravel(), post_mean-sigma_coef*post_stdev,
                     post_mean+sigma_coef*post_stdev, alpha=0.3, label='Uncertainty')
    ax1.axvline(x=fit_data_X[-1], color='green', linestyle='--', linewidth=3)
    ax1.set_title(title_left, fontsize=font_size)


    ax2.scatter(fit_data_X, obs_data_Y, c='black')
    ax2.plot(predict_candidates_X, post_mean_after, linewidth=2.5, color='black')
    ax2.fill_between(predict_candidates_X.ravel(), post_mean_after-sigma_coef*post_stdev_after, 
                     post_mean_after+sigma_coef*post_stdev_after, alpha=0.3)
    ax2.scatter(fit_data_X[-1], obs_data_Y[-1], c='red')
    ax2.axvline(x=max_stdev_design, color='green', linestyle='--', linewidth=3)
    ax2.set_title(title_right, fontsize=font_size)

    fig.suptitle(title, fontsize=font_size)
    plt.tight_layout()
    
    return fig, (ax1, ax2)




def plot_frequency_histogram_1D(fig_size:Tuple[int, int], num_data:int, design_var:npt.NDArray, bins:int, ranges:Optional[Tuple[float, float]], 
                                xlabel:str, ylabel:str, title:str, mode:str="sum"):
    if not isinstance(fig_size, tuple):
        raise TypeError(f"fig_size should be a tuple, got the type of {type(fig_size).__name__}.")
    if any([not isinstance(fs, int) for fs in fig_size]):
        raise TypeError(f"fig_size should have integer elements.")
    if len(fig_size)!=2:
        raise ValueError(f"fig_size should be of length 2, got {len(fig_size)}.")
    if not isinstance(num_data, int):
        raise TypeError(f"num_data should be an integer value, got the type of {type(num_data).__name__}.")
    if num_data<1:
        raise ValueError(f"num_data should be a positive integer, got {num_data}.")
    if not isinstance(design_var, np.ndarray):
        raise TypeError(f"design_var should be a numpy array, got the type of {type(design_var).__name__}.")
    if design_var.ndim!=1:
        raise ValueError(f"dv1 should be a 1D array, got {design_var.ndim} dimensions.")
    if not isinstance(bins, int):
        raise TypeError(f"bins should be an integer value, got the type of {type(bins).__name__}.")
    if ranges is not None:
        if not isinstance(ranges, tuple):
            raise TypeError(f"ranges should be a tuple or None, got the type of {type(ranges).__name__}.")
        if not all([isinstance(r, float) for r in ranges]):
            raise TypeError(f"ranges should contain float elements.")
        if len(ranges)!=2:
            raise ValueError(f"ranges should be of length 2, got {len(ranges)}.")
    if not isinstance(mode, str):
        raise TypeError(f"mode should be a string value, got the type of {type(mode).__name__}.")
    if mode not in ["average", "sum"]:
        raise ValueError(f"mode should be either 'average' or 'sum', got {mode}.")
    if not isinstance(xlabel, str):
        raise TypeError(f"xlabel should be a string value, got the type of {type(xlabel).__name__}.")
    if not isinstance(ylabel, str):
        raise TypeError(f"ylabel should be a string value, got the type of {type(ylabel).__name__}.")
    if not isinstance(title, str):
        raise TypeError(f"title should be a string value, got the type of {type(title).__name__}.")

       
    figure=plt.figure(figsize=fig_size)
    ax=figure.add_subplot(1,1,1)
    hist, dv1_pos=np.histogram(design_var, bins=bins, range=ranges)

    if mode=='average':
        hist=hist/num_data


    freq_pos=0
    bin_width = (ranges[1] - ranges[0]) / bins if ranges else dv1_pos[1] - dv1_pos[0]
    bin_centers = dv1_pos[:-1] + bin_width / 2
    

    ax.bar(
        x=bin_centers,
        height=hist.ravel(),
        width=bin_width,
        bottom=freq_pos,
        align='center',
        color='skyblue',       
        alpha=0.6,             
        edgecolor='black',     
        linewidth=0.8          
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return figure, ax



def plot_frequency_histogram_2D(fig_size:Tuple[int, int], num_data:int, design_vars:npt.NDArray, 
                                bins:list[int], ranges:Optional[list[list[float]]], 
                                xlabel:str, ylabel:str, zlabel:str, title:str, mode:str='sum'):
    
    if not isinstance(fig_size, tuple):
        raise TypeError(f"fig_size should be a tuple, got the type of {type(fig_size).__name__}.")
    if any([not isinstance(fs, int) for fs in fig_size]):
        raise ValueError(f"fig_size should contain integer elements.")
    if len(fig_size)!=2:
        raise ValueError(f"fig_size should be of length 2, got {len(fig_size)}.")
    if not isinstance(num_data, int):
        raise TypeError(f"num_data should be an integer value, got the type of {type(num_data).__name__}.")
    if not isinstance(design_vars, np.ndarray):
        raise TypeError(f"design_vars should be a numpy array, got the type of {type(design_vars).__name__}.")
    if design_vars.ndim!=2:
        raise ValueError(f"design_vars should be a 2D array, got {design_vars.ndim} dimensions.")
    if design_vars.shape[1]!=2:
        raise ValueError(f"design_vars should have two columns, got {design_vars.shape[1]} columns.")
    if not isinstance(bins, list):
        raise TypeError(f"bins should be a list, got the type of {type(bins).__name__}.")
    if not all([isinstance(b, int) for b in bins]):
        raise TypeError(f"bins should contain integer values.")
    if len(bins)!=2:
        raise ValueError(f"bins should be of length 2, got {len(bins)}.")
    if ranges is not None:
        if not isinstance(ranges, list):
            raise TypeError(f"ranges should be a list, got the type of {type(ranges).__name__}.")
        if not all([isinstance(r, list) for r in ranges]):
            raise TypeError(f"ranges should contain list elements.")
        if not all([isinstance(r1, float) for r1 in ranges[0]]):
            raise TypeError(f"ranges[0] should contain float type elements.")
        if not all([isinstance(r2, float) for r2 in ranges[1]]):
            raise TypeError(f"ranges[1] should contain float type elements.")
        if len(ranges)!=2:
            raise ValueError(f"ranges should contain two list elements, got {len(ranges)}.")
        if len(ranges[0])!=2:
            raise ValueError(f"ranges[0] should contain two float elements, got {len(ranges[0])}.")
        if len(ranges[1])!=2:
            raise ValueError(f"ranges[1] should contain two float elements, got {len(ranges[1])}.")
    
    if not isinstance(xlabel, str):
        raise TypeError(f"xlabel should be a string value, got the type of {type(xlabel).__name__}.")
    if not isinstance(ylabel, str):
        raise TypeError(f"ylabel should be a string value, got the type of {type(ylabel).__name__}.")
    if not isinstance(zlabel, str):
        raise TypeError(f"zlabel should be a string value, got the type of {type(zlabel).__name__}.")
    if not isinstance(title, str):
        raise TypeError(f"title should be a string value, got the type of {type(title).__name__}.")
    if not isinstance(mode, str):
        raise TypeError(f"mode should be a string value, got the type of {type(mode).__name__}.")
    if mode not in ["average", "sum"]:
        raise ValueError(f"mode should be either 'average' or 'sum', got {mode}.")

       
    figure=plt.figure()
    ax=figure.add_subplot(projection='3d')
    hist, dv1_edge, dv2_edge=np.histogram2d(design_vars[:,0], design_vars[:,1], bins=bins, range=ranges)
    if mode=="average":
        hist=hist/num_data

    dv1_pos, dv2_pos=np.meshgrid(dv1_edge[:-1], dv2_edge[:-1], indexing="ij")
    dv1_pos=dv1_pos.ravel()
    dv2_pos=dv2_pos.ravel()
    hist_pos=0

    bin_width=(ranges[0][1]-ranges[0][0])/bins[0] if ranges else dv1_edge[1] - dv1_edge[0]
    bin_depth=(ranges[1][1]-ranges[1][0])/bins[1] if ranges else dv2_edge[1] - dv1_edge[0]
    h=hist.ravel()
    
    bin_centers_dv1 = dv1_pos[:-1] + bin_width/2
    bin_centers_dv2 = dv2_pos[:-1] + bin_depth/2
    
    
    ax.bar3d(x=bin_centers_dv1, y=bin_centers_dv2, z=hist_pos, 
             dx=bin_width, dy=bin_depth, dz=h, zsort='average')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)

    return figure, ax
    





