from adopy import Task, Model, Engine
import numpy as np
from scipy.stats import norm

def mixed_log_linear(a, b, lam, given_number, N_MAX):
    return a * ((1 - lam) * given_number + lam * N_MAX / np.log(N_MAX) * np.log(given_number)) + b

def generate_grid_params():
    return {
        'a': np.arange(0.5, 1.1, 0.1),
        'b': np.arange(0, 101, 10),
        'lam': np.arange(0.1, 1.1, 0.2),
        'sigma': np.arange(1, 11, 1)
    }

def generate_grid_designs():
    return {'given_number': np.arange(5, 501, 1)}

def generate_grid_response():
    return {'response': np.arange(5, 501, 5)}

def log_likelihood(response, given_number, a, b, lam, sigma):
    N_MAX = 500

    x = given_number
    y = response

    # 예측값
    pred = a * ((1 - lam) * x + lam * N_MAX / np.log(N_MAX) * np.log(x)) + b

    # 정규분포 기반 log-likelihood
    return norm(loc=pred, scale=sigma).logpdf(y)