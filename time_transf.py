# time_transf.py
import numpy as np

# --- Constants for the piecewise function ---
# Transition points in seconds.
# T1: The first point where the behavior changes from logarithmic to linear.
T1 = 86400.0  # 1 day in seconds
# T2: The second point where the behavior changes from linear to square root.
T2 = 86400.0 * 30  # 30 days in seconds

# --- Auto-computed Parameters for a Smooth, Continuous Function ---

# We define the function in three parts and compute parameters to ensure
# the value and the first derivative (slope) match at the transition points,
# creating a seamless "stitch" between the function pieces.

# PART 1: Logarithmic behavior for t <= T1
# Function: f(t) = log(1 + t)
# We need the value and slope at the end of this interval (at T1).
Y1 = np.log1p(T1)
SLOPE1 = 1.0 / (1.0 + T1)

# PART 2: Linear behavior for T1 < t <= T2
# Function: g(t) = A*t + B
# We set its starting slope and value to match the end of the log part.
A = SLOPE1
B = Y1 - A * T1
# We calculate the function's value at the end of this linear interval (at T2).
Y2 = A * T2 + B

# PART 3: Square root behavior for t > T2
# Function: h(t) = C*sqrt(t) + D
# We set its starting slope and value to match the end of the linear part.
# The derivative h'(t) = C * 0.5 * t^(-0.5).
# h'(T2) must equal the linear slope, A. So, C * 0.5 * T2**(-0.5) = A.
C = 2 * A * np.sqrt(T2)
D = Y2 - C * np.sqrt(T2)


def transform_time(time_in_seconds):
    """
    Applies a piecewise transformation to time data (expected in seconds).

    The function has three distinct behaviors:
    1.  **Logarithmic (`log(1+t)`):** For t <= 1 day. Sensitive to small changes.
    2.  **Linear:** For 1 day < t <= 30 days. Provides stable, steady growth.
    3.  **Square Root:** For t > 30 days. Growth slows down, reducing the impact of extreme outliers.

    The function is continuous and smooth at all transition points.

    Args:
        time_in_seconds (float or np.ndarray): Time value(s) in seconds.

    Returns:
        float or np.ndarray: The transformed time value(s).
    """
    t = np.asarray(time_in_seconds, dtype=float)

    # Define the conditions and corresponding function choices.
    # np.select evaluates conditions in order and picks the first choice that is True.
    # The 'default' is used when no conditions are met.
    condlist = [
        t <= T1,  # Condition for the log part
        t > T2  # Condition for the sqrt part
    ]
    choicelist = [
        np.log1p(t),  # Choice 1: Logarithmic function
        C * np.sqrt(t) + D  # Choice 2: Square root function
    ]
    # If neither of the above is true, the time is between T1 and T2.
    default = A * t + B  # Default: Linear function

    return np.select(condlist, choicelist, default=default)


def inverse_transform_time(transformed_time):
    """
    Applies the inverse of the piecewise time transformation.
    This function converts the transformed values back to the original scale (seconds).

    Args:
        transformed_time (float or np.ndarray): The transformed time value(s).

    Returns:
        float or np.ndarray: The time value(s) in the original scale (seconds).
    """
    y = np.asarray(transformed_time, dtype=float)

    # The conditions are now based on the transformed values Y1 and Y2.
    condlist = [
        y <= Y1,  # Condition for the inverse of log part
        y > Y2  # Condition for the inverse of sqrt part
    ]
    choicelist = [
        np.expm1(y),  # Choice 1: Inverse of log is exp(y) - 1
        ((y - D) / C) ** 2  # Choice 2: Inverse of sqrt
    ]
    # If neither is true, the value corresponds to the linear part.
    default = (y - B) / A  # Default: Inverse of linear

    result = np.select(condlist, choicelist, default=default)

    # Ensure no negative time predictions due to floating point inaccuracies.
    return np.maximum(0, result)
