import numpy as np

import cicero

NUM_SIMULATIONS = 100000
MAX_STEPS = 100
CONVERGENCE_WINDOW_WIDTH = 30
DISPLAY_FPS = 0

gol = cicero.tf_game_of_life.TFToroidalGOL.from_random_states(num_simulations=NUM_SIMULATIONS)
gol.simulate(num_steps=MAX_STEPS)
cnv_properties = gol.classify_convergence(convergence_window_width=CONVERGENCE_WINDOW_WIDTH)

print("Number of constant convergences:", NUM_SIMULATIONS - cnv_properties["dynamic"].sum())
print("Number of  dynamic convergences:", cnv_properties["dynamic"].sum())

print("Dumping attractors.npy...")
np.save("attractors.npy", cnv_properties["indicator_states"])
