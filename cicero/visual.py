"""Visualization of runs"""

import cv2
import numpy as np
from matplotlib import axes, pyplot as plt


def plot_flattened_history(history: np.ndarray):
    history = history.astype(float)

    history = history.reshape(history.shape[0], -1)

    fig, (top, bot) = plt.subplots(2, 1, figsize=(16, 9))

    top.imshow(history.T, vmin=0, vmax=1, cmap="magma")
    bot.plot(history.sum(axis=1)[:-1], "b-", alpha=0.7)

    top.set_title("Flattened history")
    bot.set_title("Active cell count")
    bot.set_xlim(0, history.shape[0])
    bot.grid()

    plt.tight_layout()
    plt.show()


def replay_simulation(history: np.ndarray, fps=10, repeats=10):
    for repeat in range(1, repeats+1):
        print(f"\nPlotting evolution, repeat {repeat}/{repeats}")
        for i, array in enumerate(history, start=1):
            print(f"\rSimulation progress {i / len(history):>7.2%}", end="")
            cv2.imshow("Array", array*255)
            if cv2.waitKey(1000 // fps) == 27:
                break
    print()
    cv2.destroyAllWindows()
