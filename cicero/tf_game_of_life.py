"""TensorFlow-based simulation, used to generate big number of runs in parallel"""

import tensorflow as tf


class TFToroidalGOL:

    def __init__(self, initial_states: tf.Tensor):
        self.num_simulations = len(initial_states)
        self.state = tf.Variable(initial_states)
        self.indicator_states = tf.Variable(tf.zeros([self.num_simulations, 6, 6], dtype=tf.uint8))
        self.local_sum_kernel = tf.ones([3, 3, 1, 1], dtype=tf.float32)
        self.history = None

    def initialize_history(self, num_steps: int):
        num_simulations, width, height = self.state.get_shape()
        self.history = tf.Variable(tf.cast(tf.zeros([num_steps, num_simulations, width, height]), tf.uint8))

    @classmethod
    def from_random_states(cls, num_simulations=1000, alive_probability=None):
        if alive_probability is None:
            alive_probability = tf.random.uniform(shape=[num_simulations, 1, 1])
        state = tf.random.uniform(shape=[num_simulations, 6, 6], dtype=tf.float32) < alive_probability
        return cls(tf.cast(state, tf.float32))

    @tf.function
    def pad_state(self):
        """There is no implementation for 'wrap' padding in TF, so here is an implementation."""
        padded = tf.concat([self.state[:, :, -1:], self.state, self.state[:, :, :1]], axis=2)
        padded = tf.concat([padded[:, -1:, :], padded, padded[:, :1, :]], axis=1)
        padded = padded[..., None]
        return padded

    @tf.function
    def step_simulation(self):
        state_padded = self.pad_state()
        num_neighbours = tf.nn.conv2d(state_padded, self.local_sum_kernel, strides=1, padding="VALID")[..., 0]
        num_neighbours = num_neighbours - self.state
        survivors = tf.logical_and(self.state == 1, num_neighbours == 2)
        newborns = num_neighbours == 3
        new_state = tf.logical_or(survivors, newborns)
        self.state.assign(tf.cast(new_state, tf.float32))

    def simulate(self, num_steps=100):
        print("Executing simulation")
        self.initialize_history(num_steps)
        for step in range(1, num_steps+1):
            history = tf.cast(self.state, tf.uint8)
            self.history.scatter_nd_update(indices=[step-1], updates=[history])
            self.step_simulation()
            print(f"\rProgress {step/num_steps:.2%}", end="")
        self.history = tf.transpose(self.history, (1, 0, 2, 3))
        print()

    @tf.function
    def _classify_convergence_parallelizable_part(self, recent_history: tf.Tensor, float_fuzz=1e-5):
        # Variables and tensors holding the classification results
        convergence_constants = tf.zeros(self.num_simulations, dtype=tf.uint8)

        # Working states
        num_cells = tf.reduce_sum(recent_history, axis=(2, 3))  # [num_sim, window]

        # Zero constant convergence
        mask_zero = num_cells[:, -1] == 0

        # Non-zero constant convergence
        mask_nonzero = tf.math.reduce_variance(tf.cast(num_cells, tf.float32), axis=1) < float_fuzz

        # Dynamic convergence with periodic states ( the rest ;) )
        mask_dynamic = tf.logical_not(tf.logical_or(mask_zero, mask_nonzero))

        # Populate variable with non-zero converged indicator states
        constant_indicator_states = recent_history[:, -1, ...][mask_nonzero]
        constant_indicator_indices = tf.where(mask_nonzero)
        self.indicator_states.scatter_nd_update(constant_indicator_indices, constant_indicator_states)

        # Populating dynamic indicator states is not a trivially parallelizable problem
        ...  # Solved outside, in NumPy.

        # Set constants where constant is not 0
        convergence_constants = convergence_constants + tf.cast(mask_nonzero, tf.uint8) * num_cells[:, -1]

        # Indicate which convergences were dynamic
        convergence_dynamic = tf.cast(mask_dynamic, tf.int32)

        return convergence_dynamic, convergence_constants

    def classify_convergence(self, convergence_window_width: int, float_fuzz=1e-5):
        recent_history = self.history[:, -convergence_window_width:]
        convergence_dynamic, convergence_constants = self._classify_convergence_parallelizable_part(
            recent_history, float_fuzz)
        for idx in tf.where(convergence_dynamic)[0]:
            history = recent_history[idx]
            num_cells = tf.reduce_sum(history, axis=(1, 2))
            if tf.reduce_min(num_cells) == 1:
                raise RuntimeError
            idx_minimum = tf.argmin(num_cells)
            state_minimum = history[idx_minimum]
            self.indicator_states.scatter_nd_update([idx], [state_minimum])

        result = {"dynamic": convergence_dynamic.numpy(),
                  "constants": convergence_constants.numpy(),
                  "indicator_states": self.indicator_states.numpy()}
        return result
