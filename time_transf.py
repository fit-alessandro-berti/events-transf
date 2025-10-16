# time_transf.py
import numpy as np


def transform_time(time_in_hours):
    """
    Applies a logarithmic transformation to the time data.
    Using log1p(x) = log(1+x) is numerically stable for small x.

    Args:
        time_in_hours (float or np.ndarray): Time value(s) in hours.

    Returns:
        float or np.ndarray: The transformed time value(s).
    """
    return np.log1p(time_in_hours)


def inverse_transform_time(transformed_time):
    """
    Applies the inverse of the logarithmic transformation (expm1).
    expm1(x) = exp(x) - 1, which is the inverse of log1p.

    Args:
        transformed_time (float or np.ndarray): The transformed time value(s).

    Returns:
        float or np.ndarray: The time value(s) in the original scale (hours).
    """
    return np.expm1(transformed_time)
