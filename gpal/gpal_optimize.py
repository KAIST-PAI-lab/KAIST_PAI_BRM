from sklearn.gaussian_process import GaussianProcessRegressor
import numpy as np
from typing import Callable
import numpy.typing as npt
import inspect
from gpal.gpr_fit_predict import gpr_fit, gpr_predict, next_design
import gpal.utils



def gpal_optimize(gpr:GaussianProcessRegressor, 
                  num_DVs: int, 
                  fit_data_X:npt.NDArray[np.float64], 
                  obs_data_Y:npt.NDArray[np.float64], 
                  design_candidates_specification:list[list], 
                  design_masking_function:Callable, 
                  return_stdev: bool = True, 
                  return_covar:bool = False):
    if not isinstance(gpr, GaussianProcessRegressor):
        raise TypeError(f"gpr should be a GaussianProcessRegressor instance, got {type(gpr).__name__}.")
    if not isinstance(num_DVs, int):
        raise TypeError(f"num_DVs should be an integer value, got the type of {type(num_DVs).__name__}.")
    if num_DVs<1:
        raise ValueError(f"num_DVs should be a positive integer, got {num_DVs}.")
    if not isinstance(fit_data_X, np.ndarray):
        raise TypeError(f"fit_data_X should be a numpy array, got the type of {type(fit_data_X).__name__}.")
    if fit_data_X.dtype!=np.float64:
        raise TypeError(f"fit_data_X should have a dtype of np.float64.")
    if fit_data_X.ndim!=2:
        raise ValueError(f"fit_data_X should be a 2D array, got {fit_data_X.ndim} dimensions.")
    if fit_data_X.shape[1]!=num_DVs:
        raise ValueError(f"fit_data_X should have {num_DVs} columns; got {fit_data_X.shape[1]} columns.")
    N=fit_data_X.shape[0]
    if not isinstance(obs_data_Y, np.ndarray):
        raise TypeError(f"obs_data_Y should be a numpy array, got the type of {type(obs_data_Y).__name__}.")
    if obs_data_Y.dtype!=np.float64:
        raise TypeError(f"obs_data_Y should have a dtype of np.float64.")
    if obs_data_Y.ndim!=1:
        raise ValueError(f"obs_data_Y should be a 1D array, got {obs_data_Y.ndim} dimensions.")
    if obs_data_Y.shape[0]!=N:
        raise ValueError(f"obs_data_Y should be of length {N}, got {obs_data_Y.shape[0]}.")
    if not isinstance(design_candidates_specification, list):
        raise TypeError(f"design_candidates_specification should be a list.")
    if not all([isinstance(spec, list) for spec in design_candidates_specification]):
        raise TypeError(f"design_candidates_specification should contain list elements.")
    if len(design_candidates_specification)!=num_DVs:
        raise ValueError(f"design_candidates_specification should have {num_DVs} list elements, got {len(design_candidates_specification)}.")
    if not all([len(spec)==3 for spec in design_candidates_specification]):
        raise ValueError(f"The elements of design_candidates_specification should be of length 3.")
    if not all([isinstance(spec[2], int) for spec in design_candidates_specification]):
        raise TypeError(f"The 2-th element of each list in design_candidates_specification should be an integer value.")
    if not callable(design_masking_function):
        raise TypeError(f"design_masking_function should be a callable function, got the type of {type(design_masking_function).__name__}.")
    masking_function_params_num=len(inspect.signature(design_masking_function).parameters)
    if masking_function_params_num != num_DVs:
        raise ValueError(f"design_masking_function should have {num_DVs} parameters, got {masking_function_params_num} parameters.")
    if not isinstance(return_stdev, bool):
        raise TypeError(f"return_stdev should be a bool value, got the type of {type(return_stdev).__name__}.")
    if not isinstance(return_covar, bool):
        raise TypeError(f"return_covar should be a bool value, got the type of {type(return_covar).__name__}.")    


    design_candidates=[]
    for i in range(num_DVs):
        start_value=design_candidates_specification[i][0]
        end_value=design_candidates_specification[i][1]
        interval_value=design_candidates_specification[i][2]
        candidates_num_per_axis=int((end_value-start_value)/interval_value)+1
        design_candidates.append(np.linspace(start_value, end_value, candidates_num_per_axis))

    design_candidates_grid=list(np.meshgrid(*design_candidates, indexing='ij'))

    design_candidate_coordinates=np.stack(design_candidates_grid, -1).T
    design_mask_binary=design_masking_function(*design_candidate_coordinates)  # Masking the 2D coordinates with a lambda equation
    predict_candidates_X=design_candidate_coordinates[:,design_mask_binary].T
    log_marginal_likelihood=gpr_fit(gpr=gpr, num_DVs=num_DVs, fit_data_X=fit_data_X, obs_data_Y=obs_data_Y)
    posterior_prediction=gpr_predict(gpr, num_DVs=num_DVs, predict_candidates_X=predict_candidates_X,
                                    return_stdev=return_stdev, return_covar=return_covar)
    next_design_coordinate, post_mean, post_stdev=next_design(posterior_prediction=posterior_prediction, num_DVs=num_DVs,
                                                            predict_candidates_X=predict_candidates_X)

    return next_design_coordinate, post_mean, post_stdev, log_marginal_likelihood



'''
def gpal_optimize2D(gpr:GaussianProcessRegressor, dv1:npt.NDArray[np.float64], dv2:npt.NDArray[np.float64], 
                  est:npt.NDArray[np.float64], dv1Spec:list, dv2Spec:list, 
                  masking:Callable, r_s: bool = True, r_c:bool = False):
    if not isinstance(gpr, GaussianProcessRegressor):
        raise TypeError(f"gpr should be a GaussianProcessRegressor instance, got {type(gpr).__name__}.")
    if not isinstance(dv1, np.ndarray):
        raise TypeError(f"dv1 should be a numpy array.")
    if dv1.dtype!=np.float64:
        raise TypeError(f"dv1 should have a dtype of np.float64.")
    if dv1.ndim!=2 or dv1.shape[1]!=1:
        raise ValueError(f"dv1 should be a 2D array with a single column, got the shape of {dv1.shape}.")
    N=dv1.shape[0]
    if not isinstance(dv2, np.ndarray):
        raise TypeError(f"dv2 should be a numpy array.")
    if dv2.dtype!=np.float64:
        raise TypeError(f"dv2 should have a dtype of np.float64.")
    if dv2.ndim!=2 or dv2.shape[1]!=1:
        raise ValueError(f"dv2 should be a 2D array with a single column, got the shape fo {dv2.shape}.")
    if dv2.shape[0]!=N:
        raise ValueError(f"dv2 should be of length {N}, got {dv2.shape[0]}.")
    if not isinstance(est, np.ndarray):
        raise TypeError(f"est should be a numpy array.")
    if est.dtype!=np.float64:
        raise TypeError(f"est should have a dtype of np.float64.")
    if est.ndim!=1:
        raise ValueError(f"ests should be a 1D array.")
    if est.shape[0]!=N:
        raise ValueError(f"ests should be of length {N}, got {est.shape[0]}.")
    if not isinstance(dv1Spec, list):
        raise TypeError(f"dv1Spec should be a list.")
    if len(dv1Spec)!=3:
        raise ValueError(f"dv1Spec should have a length of 3, got {len(dv1Spec)}.")
    if not isinstance(dv1Spec[2], int):
        raise TypeError(f"The last element of dv1Spec should be an integer value.")
    if not isinstance(dv2Spec, list):
        raise TypeError(f"dv2Spec should be a list.")
    if len(dv2Spec)!=3:
        raise ValueError(f"dv2Spec should have a length of 3, got {len(dv2Spec)}.")
    if not isinstance(dv2Spec[2], int):
        raise TypeError(f"The last element of dv2Spec should be an integer value.")
    if not callable(masking):
        raise TypeError(f"masking should be a callable function.")
    if not isinstance(r_s, bool):
        raise TypeError(f"r_s should be a bool value, got {type(r_s).__name__}.")
    if not isinstance(r_c, bool):
        raise TypeError(f"r_c should be a bool value, got {type(r_c).__name__}.")    
    
    X_fit=np.column_stack([dv1, dv2])
    y_fit=est

    dv1_cands=np.linspace(dv1Spec[0], dv1Spec[1], dv1Spec[2])
    dv2_cands=np.linspace(dv2Spec[0], dv2Spec[1], dv2Spec[2])

    dv1_grid, dv2_grid=np.meshgrid(dv1_cands, dv2_cands, indexing='ij')
    dv1_grid=dv1_grid.ravel()
    dv2_grid=dv2_grid.ravel()

    mask=masking(dv1_grid, dv2_grid)  # Masking the 2D coordinates with a lambda equation
    dvs_grid=np.column_stack([dv1_grid, dv2_grid])
    X_pred=dvs_grid[mask,:]
    lml=gprFit2D(gpr, X_fit, y_fit)
    pred=gprPredict2D(gpr, X_pred, r_s, r_c)
    nextX, pMean, pStd=nextDesign2D(pred, X_pred)


    return nextX, pMean, pStd, lml




def gpal_optimize1D(gpr:GaussianProcessRegressor, dv1:npt.NDArray[np.float64], 
                  est:npt.NDArray[np.float64], dv1Spec:list,
                  masking:Callable, r_s: bool = True, r_c:bool = False):
    if not isinstance(gpr, GaussianProcessRegressor):
        raise TypeError(f"gpr should be a GaussianProcessRegressor instance, got {type(gpr).__name__}.")
    if not isinstance(dv1, np.ndarray):
        raise TypeError(f"dv1 should be a numpy array.")
    if dv1.dtype!=np.float64:
        raise TypeError(f"dv1 should have a dtype of np.float64.")
    if dv1.ndim!=2 or dv1.shape[1]!=1:
        raise ValueError(f"dv1 should be a 2D numpy array with a single column, got the shape of {dv1.shape}.")
    N=dv1.shape[0]
    if not isinstance(est, np.ndarray):
        raise TypeError(f"est should be a numpy array.")
    if est.dtype!=np.float64:
        raise TypeError(f"est should have a dtype of np.float64.")
    if est.ndim!=1:
        raise ValueError(f"ests should be a 1D array.")
    if est.shape[0]!=N:
        raise ValueError(f"ests should be of length {N}, got {est.shape[0]}.")
    if not isinstance(dv1Spec, list):
        raise TypeError(f"dv1Spec should be a list.")
    if len(dv1Spec)!=3:
        raise ValueError(f"dv1Spec should have a length of 3, got {len(dv1Spec)}.")
    if not isinstance(dv1Spec[2], int):
        raise TypeError(f"The last element of dv1Spec should be an integer value.")
    if not callable(masking):
        raise TypeError(f"masking should be a callable function.")
    if not isinstance(r_s, bool):
        raise TypeError(f"r_s should be a bool value, got {type(r_s).__name__}.")
    if not isinstance(r_c, bool):
        raise TypeError(f"r_c should be a bool value, got {type(r_c).__name__}.")    
    
    X_fit=dv1
    y_fit=est

    dv1_grid=np.linspace(dv1Spec[0], dv1Spec[1], dv1Spec[2])

    mask=masking(dv1_grid)  # Masking the 1D coordinates with a lambda equation
    X_pred=np.expand_dims(dv1_grid[mask], -1)

    lml=gprFit1D(gpr, X_fit, y_fit)
    pred=gprPredict1D(gpr, X_pred, r_s, r_c)
    nextX, pMean, pStd=nextDesign1D(pred, X_pred)


    return nextX, pMean, pStd, lml

'''


