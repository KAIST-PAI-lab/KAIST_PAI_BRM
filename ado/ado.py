from adopy import Task, Model, Engine
import numpy as np
from scipy.stats import norm

def mixed_log_linear(a, b, lam, given_number, N_MAX):
    return a * ((1 - lam) * given_number + lam * N_MAX / np.log(N_MAX) * np.log(given_number)) + b

def generate_grid_params():
    return {
        'a': np.linspace(0.5, 1, 11),
        'b': np.linspace(0, 100, 11),
        'lam': np.linspace(0, 1, 21),
        'sigma': np.linspace(1, 100, 11)
    }

def generate_grid_designs():
    return {'given_number': np.arange(5, 501, 5)}

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