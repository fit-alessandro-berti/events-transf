import numpy as np
def transform_time (time_in_hours ):
    return np .log1p (time_in_hours )
def inverse_transform_time (transformed_time ):
    return np .expm1 (transformed_time )
