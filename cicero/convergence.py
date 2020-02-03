"""
Types and helpers related to convergence and convergence analysis
These implementations were mostly used in devising heuristics for convergence detection.
"""

import dataclasses

import numpy as np

NONE = "none"
CONSTANT = "constant"
DYNAMIC = "dynamic"
TYPES = [NONE, CONSTANT, DYNAMIC]


@dataclasses.dataclass
class ConvergenceProperties:
    type: str = ""
    constant: int = -1
    step: int = -1
    indicator_state: np.ndarray = None

    def set_indicator_state(self, state_history: np.ndarray):
        """Indicator state is the state which is used to represent a convergence"""
        if self.type == CONSTANT:
            self.indicator_state = state_history[-1]  # Simply take a state after convergence
            assert np.sum(self.indicator_state) == self.constant
            return

        # Take the state from recent history with the least number of cells
        indicator_index = np.argmin(state_history.sum(axis=(1, 2)))
        self.indicator_state = state_history[indicator_index]

    def set_step(self, step):
        self.step = step


def calculate_oscillation_statistics(far_past: np.ndarray, recent_past: np.ndarray):
    """
    Oscillation is detected by partinioning the detection window into 2 equal parts,
    and comparing means and variances
    """
    mean_far_past = np.mean(far_past)
    mean_recent_past = np.mean(recent_past)
    var_far_past = np.var(far_past)
    var_recent_past = np.var(recent_past)
    windowed_mean_difference = np.abs(mean_far_past - mean_recent_past)
    windowed_var_difference = np.abs(var_far_past - var_recent_past)
    return windowed_mean_difference, windowed_var_difference


def check(last_num_cells: np.ndarray) -> ConvergenceProperties:

    last_num_cells = last_num_cells.astype(int)

    n = len(last_num_cells)
    half_n = n // 2

    far_past = last_num_cells[:half_n]
    recent_past = last_num_cells[half_n:]

    if last_num_cells[-1] == 0:  # Converged to 0
        return ConvergenceProperties(CONSTANT, 0)
    if len(last_num_cells) < 5:  # Too early to check for non-zero convergence
        return ConvergenceProperties(NONE)

    if np.isclose(np.var(recent_past), 0):  # Converged to a sequence of non-zero states
        return ConvergenceProperties(CONSTANT, last_num_cells[-1])

    if len(last_num_cells) < 20:
        return ConvergenceProperties(NONE)  # Too early to check for periodicity

    windowed_mean_difference, windowed_var_difference = calculate_oscillation_statistics(far_past, recent_past)

    # Magic number was determined empirically
    if windowed_mean_difference < 0.2 or windowed_var_difference < 0.2:  # Converged to a dynamic attractor
        return ConvergenceProperties(DYNAMIC)

    return ConvergenceProperties(NONE)
