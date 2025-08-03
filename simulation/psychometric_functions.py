# Collection of psychometric models and their likelihood functions

import numpy as np


def mixed_log_linear(number_array, parameter_list, N_MAX):
    a = parameter_list[0]
    b = parameter_list[1]
    lam = parameter_list[2]

    number_estimate = (
        a
        * (
            (1 - lam) * number_array
            + lam * N_MAX / np.log(N_MAX) * np.log(number_array)
        )
        + b
    )

    return number_estimate


def mixed_log_linear_likelihood(parameters, number_array, simulated_estimate, N_MAX):
    a = parameters[0]
    b = parameters[1]
    parameter_lambda = parameters[2]
    sigma = parameters[3]

    number_estimate = (
        a
        * (
            (1 - parameter_lambda) * number_array
            + parameter_lambda * N_MAX / np.log(N_MAX) * np.log(number_array)
        )
        + b
    )
    L = sum(
        np.log(sigma)
        + 0.5 * np.log(2 * np.pi)
        + ((simulated_estimate - number_estimate) ** 2) / (2 * sigma**2)
    )
    return L


def one_cyclic_power(number_array, parameter_list, N_MAX):
    beta = parameter_list[0]

    number_estimate = N_MAX * (
        number_array**beta / (number_array**beta + (N_MAX - number_array) ** beta)
    )

    return number_estimate


def one_cyclic_power_likelihood(estimate, given_num, beta, sigma):
    N_MAX = 100
    number_estimate = N_MAX * (
        given_num**beta / (given_num**beta + (N_MAX - given_num) ** beta)
    )

    L = (
        np.log(sigma)
        + 0.5 * np.log(2 * np.pi)
        + ((estimate - number_estimate) ** 2) / (2 * sigma**2)
    )

    return L


def two_cyclic_power(number_array, parameter_list, N_MAX):
    beta = parameter_list[0]
    mid = N_MAX / 2

    number_array = np.asarray(number_array, dtype=float)
    number_estimate = np.empty_like(number_array)

    # Left half (0 – N_MAX/2)
    left_mask = number_array <= mid
    x_left = number_array[left_mask]
    number_estimate[left_mask] = (
        mid * (x_left**beta) / (x_left**beta + (mid - x_left) ** beta)
    )

    # Right half (N_MAX/2 – N_MAX)
    right_mask = ~left_mask
    x_right = number_array[right_mask] - mid
    number_estimate[right_mask] = mid + mid * (x_right**beta) / (
        x_right**beta + (N_MAX - mid - x_right) ** beta
    )

    return number_estimate
