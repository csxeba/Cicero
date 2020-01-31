import numpy as np

NONE = "none"
CONSTANT = "constant"
OSCILLATORY = "oscillatory"
TYPES = [NONE, CONSTANT, OSCILLATORY]


def calculate_oscillation_statistics(far_past: np.ndarray, recent_past: np.ndarray):
    mean_far_past = np.mean(far_past)
    mean_recent_past = np.mean(recent_past)
    var_far_past = np.var(far_past)
    var_recent_past = np.var(recent_past)
    windowed_mean_difference = np.abs(mean_far_past - mean_recent_past)
    windowed_var_difference = np.abs(var_far_past - var_recent_past)
    return windowed_mean_difference, windowed_var_difference


def check(last_num_cells: np.ndarray):

    n = len(last_num_cells)
    half_n = n // 2

    far_past = last_num_cells[:half_n]
    recent_past = last_num_cells[half_n:]

    if last_num_cells[-1] == 0:
        return CONSTANT, 0, 0, 0
    if len(last_num_cells) < 5:
        return NONE, None, 0, 0

    diffs = np.diff(recent_past)
    if np.isclose(np.var(diffs), 0):
        return CONSTANT, last_num_cells[-1], 0, 0

    if len(last_num_cells) < 20:
        return NONE, None, 0, 0  # Too early to check for periodicity

    windowed_mean_difference, windowed_var_difference = calculate_oscillation_statistics(far_past, recent_past)

    if windowed_mean_difference < 0.2 or windowed_var_difference < 0.2:
        return OSCILLATORY, None, windowed_mean_difference, windowed_var_difference

    return NONE, None, windowed_mean_difference, windowed_var_difference
