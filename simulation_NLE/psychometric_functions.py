# Collection of psychometric models and their likelihood functions
from pathlib import Path

import numpy as np
import yaml

sim_config_path = Path(__file__).parent / "simulation_config.yaml"
with sim_config_path.open(encoding="utf-8") as f:
    SIM_CONFIG = yaml.safe_load(f)

global N_MAX
N_MAX = SIM_CONFIG["parameter_lists"]["N_MAX"]


def mixed_log_linear(param_slope, param_intercept, param_log_mix, given_number):
    """_summary_
    Args:
        param_slope (_type_): between 0 and 1
        param_intercept (_type_): between 0 and N_MAX
        param_log_mix (_type_): between 0 and 1
        given_number (_type_): _description_

    Returns:
        _type_: _description_
    """

    number_estimate = (
        param_slope
        * (
            (1 - param_log_mix) * given_number
            + param_log_mix * N_MAX / np.log(N_MAX) * np.log(given_number)
        )
        + param_intercept
    )

    return number_estimate


def mixed_log_linear_likelihood(
    param_slope,
    param_intercept,
    param_log_mix,
    param_noise,
    given_number,
    simulated_estimate,
):

    number_estimate = (
        param_slope
        * (
            (1 - param_log_mix) * given_number
            + param_log_mix * N_MAX / np.log(N_MAX) * np.log(given_number)
        )
        + param_intercept
    )

    L = (
        np.log(param_noise)
        + 0.5 * np.log(2 * np.pi)
        + ((simulated_estimate - number_estimate) ** 2) / (2 * param_noise**2)
    )

    return -L


def mixed_log_linear_likelihood_scipy_minimize(
    parameters,
    given_number,
    simulated_estimate,
):
    param_slope, param_intercept, param_log_mix, param_noise = (
        parameters[0],
        parameters[1],
        parameters[2],
        parameters[3],
    )

    number_estimate = (
        param_slope
        * (
            (1 - param_log_mix) * given_number
            + param_log_mix * N_MAX / np.log(N_MAX) * np.log(given_number)
        )
        + param_intercept
    )

    L = np.sum(
        np.log(param_noise)
        + 0.5 * np.log(2 * np.pi)
        + ((simulated_estimate - number_estimate) ** 2) / (2 * param_noise**2)
    )

    return L


def mixed_log_linear_generate_response(
    param_slope, param_intercept, param_log_mix, param_noise, given_number
):
    pred = (
        param_slope
        * (
            (1 - param_log_mix) * given_number
            + param_log_mix * N_MAX / np.log(N_MAX) * np.log(given_number)
        )
        + param_intercept
    )

    # noise 추가 (정규분포)
    response = np.random.normal(loc=pred, scale=param_noise)
    response = np.clip(response, 0, N_MAX)
    return response


def one_cyclic_power(param_exponent, given_number):
    number_estimate = N_MAX * (
        given_number**param_exponent
        / (given_number**param_exponent + (N_MAX - given_number) ** param_exponent)
    )

    return number_estimate


def one_cyclic_power_likelihood(
    param_exponent, param_noise, given_number, simulated_estimate
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

    return -L


def one_cyclic_power_likelihood_scipy_minimize(
    parameters, given_number, simulated_estimate
):
    param_exponent, param_noise = (
        parameters[0],
        parameters[1],
    )

    number_estimate = N_MAX * (
        given_number**param_exponent
        / (given_number**param_exponent + (N_MAX - given_number) ** param_exponent)
    )

    L = np.sum(
        np.log(param_noise)
        + 0.5 * np.log(2 * np.pi)
        + ((simulated_estimate - number_estimate) ** 2) / (2 * param_noise**2)
    )

    return L


def one_cyclic_power_generate_response(param_exponent, param_noise, given_number):
    pred = N_MAX * (
        given_number**param_exponent
        / (given_number**param_exponent + (N_MAX - given_number) ** param_exponent)
    )

    # noise 추가 (정규분포)
    response = np.random.normal(loc=pred, scale=param_noise)
    response = np.clip(response, 0, N_MAX)
    return response


def two_cyclic_power(param_exponent_left, param_exponent_right, given_number):
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

    return -L


def two_cyclic_power_likelihood_scipy_minimize(
    parameters, given_number, simulated_estimate
):
    param_exponent_left, param_exponent_right, param_noise = (
        parameters[0],
        parameters[1],
        parameters[2],
    )

    number_estimate = N_MAX * (
        (given_number**param_exponent_left)
        / (
            given_number**param_exponent_left
            + (N_MAX - given_number) ** param_exponent_right
        )
    )

    L = np.sum(
        np.log(param_noise)
        + 0.5 * np.log(2 * np.pi)
        + ((simulated_estimate - number_estimate) ** 2) / (2 * param_noise**2)
    )

    return L


def two_cyclic_power_generate_response(
    param_exponent_left, param_exponent_right, param_noise, given_number
):
    pred = N_MAX * (
        (given_number**param_exponent_left)
        / (
            given_number**param_exponent_left
            + (N_MAX - given_number) ** param_exponent_right
        )
    )

    # noise 추가 (정규분포)
    response = np.random.normal(loc=pred, scale=param_noise)
    response = np.clip(response, 0, N_MAX)
    return response
