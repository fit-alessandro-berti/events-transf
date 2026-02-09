import numpy as np


def transform_time(time_in_hours):
    t = np.asarray(time_in_hours, dtype=float)
    return np.sqrt(np.maximum(t, 0.0))


def inverse_transform_time(transformed_time):
    y = np.asarray(transformed_time, dtype=float)
    return np.maximum(y, 0.0) ** 2
