from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *
import numpy as np
from functools import reduce
from typing import Optional
import numpy.typing as npt
from nlt.gpal.gpr_fit_predict import gprFit
from nlt.gpal.gpr_fit_predict import gprPredict
from nlt.gpal.gpr_fit_predict import nextDesign
from nlt.gpal.gpr_instance import GPRInstance
from nlt.gpal.gpr_instance import KernelInstance
from nlt.gpal.utils import *


## args
## NT: The number of optimization iterations
## Nf: The batch size (length in 0-th dimension) of data for fitting
## Np: The batch size (length in 0-th diemnsion) of data for prediction
## Nk: number of combined kernels - nKernel of GPRInstance
## kti: KernelTypeIndex of argsConstructor
## pl: paramsList of argsConstructor
## mi: mulIdx of GPRInstance
## si: sumIdx of GPRInstance
## r_s: return_std
## r_c: return_cov
## alpha: alpha of GPRInstance
## n_y: normalize_y of GPRInstance
## n_r_o: n_restarts_optimizer of GPRInstance
## r_s: random_state of GPRInstance

def gpal_optimize(args):

    lmlArray=np.zeros((args.NT,))
    if args.r_s:
        stdArray=np.zeros((args.NT, args.Np))
    if args.r_c:
        covArray=np.zeros((args.NT, args.Np, args.Np))

    kernel_type_lists, parameter_dicts = argsConstructor(args.Nk, args.kti, args.pl)
    kernel_instance, gpr_instance = GPRInstance(args.Nk, kernel_type_lists, parameter_dicts, args.mi, args.si, args.alpha, args.n_y, args.n_r_o, args.r_s)
    
    # Loading the data - to be implemented
    '''
    X_fit=
    X_pred=
    y=
    '''

    lml=gprFit(gpr_instance, X_fit, y)
    preds=gprPredict(gpr_instance, X_pred, args.r_s, args.r_c)
    nextX=nextDesign(preds, X_fit)



