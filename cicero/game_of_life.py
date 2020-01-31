import json
from collections import deque

import numpy as np
from scipy import signal

from . import convergence


class ThoroidalGOL:

    def __init__(self, initial_state: np.ndarray, convergence_memory=20):
        self.initial_state = initial_state
        self.state = initial_state.copy()
        self.local_sum_kernel = np.ones([3, 3])
        self._last_num_cells = deque(maxlen=convergence_memory)
        self.convergence_type = convergence.NONE
        self.convergence_constant = None
        self.history = None
        self.debug_logs = {"wmd": [], "wvd": []}

    @classmethod
    def from_random_states(cls,
                           width: int = 6,
                           height: int = 6,
                           convergence_memory: int = 20,
                           alive_probability: float = None,
                           seed: int = None):

        if alive_probability is None:
            alive_probability = np.random.random()
        if seed is not None:
            np.random.seed(seed)
        state = (np.random.uniform(size=[width, height]) < alive_probability).astype("uint8")
        return cls(state, convergence_memory)

    @classmethod
    def from_json(cls, json_path: str, convergence_memory=20):
        state = json.load(open(json_path))
        state = np.array(state).astype("uint8")
        return cls(state, convergence_memory)

    def simulate(self, steps=1):
        history = []

        for repeat in range(steps):
            self._last_num_cells.append(self.state.sum())
            history.append(self.state.copy())
            summed = signal.convolve2d(self.state, self.local_sum_kernel, mode="same", boundary="wrap")
            num_neighbours = summed - self.state
            survivors = np.logical_and(self.state == 1, num_neighbours == 2)
            newborns = num_neighbours == 3
            new_state = np.logical_or(survivors, newborns)
            self.state = new_state.astype("uint8")
            self.convergence_type, self.convergence_constant, wmd, wvd = convergence.check(
                np.array(self._last_num_cells))
            self.debug_logs["wmd"].append(wmd)
            self.debug_logs["wvd"].append(wvd)
            if self.convergence_type != convergence.NONE:
                break
        
        self.history = np.array(history)

    def save(self, path):
        as_json = [[cell for cell in row] for row in self.initial_state.T]
        with open(path, "w") as handle:
            json.dump(as_json, handle)
