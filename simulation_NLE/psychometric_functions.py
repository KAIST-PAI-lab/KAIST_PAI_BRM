# Collection of psychometric models and their likelihood functions

import numpy as np


def mixed_log_linear(param_slope, param_bias, param_log_mix, given_number, N_MAX):
    """_summary_
    Args:
        param_slope (_type_): between 0 and 1
        param_bias (_type_): between 0 and N_MAX
        param_log_mix (_type_): between 0 and 1
        given_number (_type_): _description_
        N_MAX (_type_): _description_

    Returns:
        _type_: _description_
    """

    number_estimate = (
        param_slope
        * (
            (1 - param_log_mix) * given_number
            + param_log_mix * N_MAX / np.log(N_MAX) * np.log(given_number)
        )
        + param_bias
    )

    return number_estimate


def mixed_log_linear_likelihood(
    param_slope,
    param_intercept,
    param_log_mix,
    param_noise,
    given_number,
    simulated_estimate,
    N_MAX,
):
    number_estimate = (
        param_slope
        * (
            (1 - param_log_mix) * given_number
            + param_log_mix * N_MAX / np.log(N_MAX) * np.log(given_number)
        )
        + param_intercept
    )
    L = sum(
        np.log(param_noise)
        + 0.5 * np.log(2 * np.pi)
        + ((simulated_estimate - number_estimate) ** 2) / (2 * param_noise**2)
    )
    return L


def one_cyclic_power(param_exponent, given_number, N_MAX):
    number_estimate = N_MAX * (
        given_number
        * param_exponent
        / (given_number * param_exponent + (N_MAX - given_number) ** param_exponent)
    )

    return number_estimate


def one_cyclic_power_likelihood(
    param_exponent, param_noise, given_number, simulated_estimate, N_MAX
):

    number_estimate = N_MAX * (
        given_number**param_exponent
        / (given_number**param_exponent + (N_MAX - given_number) ** param_exponent)
    )

    L = (
        np.log(param_noise)
        + 0.5 * np.log(2 * np.pi)
        + ((simulated_estimate - number_estimate) ** 2) / (2 * param_noise**2)
    )

    return L


def two_cyclic_power(param_exponent_left, param_exponent_right, given_number, N_MAX):
    number_estimate = N_MAX * (
        (given_number**param_exponent_left)
        / (
            given_number**param_exponent_left
            + (N_MAX - given_number) ** param_exponent_right
        )
    )
    return number_estimate


def two_cyclic_power_likelihood(
    param_exponent_left,
    param_exponent_right,
    param_noise,
    given_number,
    simulated_estimate,
    N_MAX,
):
    number_estimate = N_MAX * (
        (given_number**param_exponent_left)
        / (
            given_number**param_exponent_left
            + (N_MAX - given_number) ** param_exponent_right
        )
    )

    L = (
        np.log(param_noise)
        + 0.5 * np.log(2 * np.pi)
        + ((simulated_estimate - number_estimate) ** 2) / (2 * param_noise**2)
    )

    return L
