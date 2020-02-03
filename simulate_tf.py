import argparse

import numpy as np

import cicero

parser = argparse.ArgumentParser("Cicero - TensorFlow simulation", description="Toroidal Game of Life")

parser.add_argument("--num-simulations", default=100000, type=int)
parser.add_argument("--max-steps", default=100, type=int)
parser.add_argument("--convergence-detector-window", default=30, type=int)

arg = parser.parse_args()

gol = cicero.tf_game_of_life.TFToroidalGOL.from_random_states(num_simulations=arg.num_simulations)
gol.simulate(num_steps=arg.max_steps)
cnv_properties = gol.classify_convergence(convergence_window_width=arg.convergence_detector_window)

print("Dumping to attractors.npy...")
np.save("attractors.npy", cnv_properties["indicator_states"])
