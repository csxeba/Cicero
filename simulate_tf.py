import argparse

import numpy as np

import cicero

descr = """Toroidal Game of Life.

Use this script to run simulations in parallel.
All simulations will produce an attractor candidate,
which will be simply dumped to attractors.npy.

This can be picked up by analyze_attractors.py, which
will answer all homework questions.
"""

parser = argparse.ArgumentParser("Cicero - TensorFlow simulation",
                                 description=descr)

parser.add_argument("--num-simulations", default=100000, type=int,
                    help="Number of parallel simulations")
parser.add_argument("--max-steps", default=100, type=int,
                    help="How many steps to run all simulations for")
parser.add_argument("--convergence-detector-window", default=30, type=int,
                    help="Slice size, on which convergence will be determined")

arg = parser.parse_args()

gol = cicero.tf_game_of_life.TFToroidalGOL.from_random_states(num_simulations=arg.num_simulations)
gol.simulate(num_steps=arg.max_steps)
cnv_properties = gol.classify_convergence(convergence_window_width=arg.convergence_detector_window)

print("Dumping to attractors.npy...")
np.save("attractors.npy", cnv_properties["indicator_states"])
