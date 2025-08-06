"""
The 1-dimensional number line task.

In this task, a participant is presented with a number and is asked to estimate its position on a line.
"""
import numpy as np
from adopy.base import Engine, Task, Model

__all__ = ['TaskNL', 'ModelLogLinear', 'EngineNL']


class TaskNL(Task):
    """
    The Task class for the 1D number line task.

    Design variables
        - ``number_array`` - The number presented to the participant.
        - ``N_MAX`` - The maximum value on the number line.

    Responses
        - ``simulated_estimate`` - The participant's estimate of the number's position.
    """

    def __init__(self):
        super(TaskNL, self).__init__(
            name='Number Line Task',
            designs=['number_array', 'N_MAX'],
            responses=['simulated_estimate']
        )


class ModelLogLinear(Model):
    r"""
    The log-linear model for the number line task.

    .. math::
        estimate = a \cdot ((1 - \lambda) \cdot n + \lambda \cdot \frac{N_{MAX}}{\ln(N_{MAX})} \cdot \ln(n)) + b

    Model parameters
        - ``a``
        - ``b``
        - ``parameter_lambda``
        - ``sigma`` - standard deviation of the noise
    """

    def __init__(self):
        super(ModelLogLinear, self).__init__(
            name='Log-linear model for the NL task',
            task=TaskNL(),
            params=['a', 'b', 'parameter_lambda', 'sigma']
        )

    def log_linear(self, number_array, a, b, parameter_lambda, N_MAX):
        return a * ((1 - parameter_lambda) * number_array + parameter_lambda * N_MAX / np.log(N_MAX) * np.log(number_array)) + b

    def compute(self, number_array, simulated_estimate, N_MAX, a, b, parameter_lambda, sigma):
        number_estimate = self.log_linear(number_array, a, b, parameter_lambda, N_MAX)
        L = np.sum(np.log(sigma) + 0.5 * np.log(2 * np.pi) + ((simulated_estimate - number_estimate) ** 2) / (2 * sigma ** 2))
        return L


class EngineNL(Engine):
    """
    The Engine class for the NL task.
    """

    def __init__(self, model, grid_design, grid_param, **kwargs):
        if not isinstance(model.task, TaskNL):
            raise RuntimeError(
                'The model should be implemented for the NL task.')

        # Assuming a grid for the response, this might need adjustment
        grid_response = {'simulated_estimate': np.arange(1, 101)}

        super(EngineNL, self).__init__(
            task=model.task,
            model=model,
            grid_design=grid_design,
            grid_param=grid_param,
            grid_response=grid_response,
            **kwargs,
        )
