import json
from collections import deque

import numpy as np
from scipy import signal

from . import convergence


class ToroidalGOL:

    """
    NumPy based simulation, less parallelism, better traceability.
    It was used to find heuristics for convergence determination.
    """

    def __init__(self, initial_state: np.ndarray, convergence_memory=20):
        self.initial_state = initial_state
        self.state = initial_state.copy()
        self.local_sum_kernel = np.ones([3, 3])
        self.num_cells = deque(maxlen=convergence_memory)
        self.convergence_properties = None
        self.history = None

    @classmethod
    def from_random_state(cls,
                          width: int = 6,
                          height: int = 6,
                          convergence_memory: int = 20,
                          alive_probability: float = None,
                          seed: int = None):

        if not alive_probability:
            alive_probability = np.random.random()
        if seed is not None:
            np.random.seed(seed)
        state = (np.random.uniform(size=[width, height]) < alive_probability).astype("uint8")
        return cls(state, convergence_memory)

    @classmethod
    def from_json(cls, json_path: str, convergence_memory=20) -> "ToroidalGOL":
        state = json.load(open(json_path))
        state = np.array(state).astype("uint8")
        return cls(state, convergence_memory)

    def simulate(self, steps=1, break_on_convergence=True, verbose=0):
        history = []

        for step in range(steps):

            # Keep a running log of past number of cells
            self.num_cells.append(self.state.sum())

            # We save all the states
            history.append(self.state.copy())

            # GOL logic is implemented as a convolution for speed and code compactness
            summed = signal.convolve2d(self.state, self.local_sum_kernel, mode="same", boundary="wrap")
            num_neighbours = summed - self.state
            survivors = np.logical_and(self.state == 1, num_neighbours == 2)
            newborns = num_neighbours == 3
            new_state = np.logical_or(survivors, newborns)
            self.state = new_state.astype("uint8")

            # Check for convergence
            convergence_props = convergence.check(np.array(self.num_cells))

            if convergence_props.type != convergence.NONE:
                if self.convergence_properties is None:
                    convergence_props.step = step
                    self.convergence_properties = convergence_props
                if break_on_convergence:
                    if verbose:
                        print("\nSimulation shut down early due to convergence:")
                        print(f"CONVERGENCE TYPE: <{convergence_props.type}> "
                              f"CONVERGENCE CONSTANT: {convergence_props.constant} "
                              f"CONVERGED IN STEP <{step}>")
                    break
        
        self.history = np.array(history)
