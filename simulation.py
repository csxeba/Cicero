from collections import defaultdict
from pprint import pprint

import numpy as np
import cv2
from matplotlib import pyplot as plt, axes

from cicero.game_of_life import ThoroidalGOL
from cicero import convergence


def plot_history(gol: ThoroidalGOL):
    top: axes.Axes
    bot: axes.Axes

    history = gol.history.astype(float)

    history = history.reshape(history.shape[0], -1)
    diffs = np.sum(np.abs(history[:-1] - history[1:]), axis=1)
    fig, (top, bot) = plt.subplots(2, 1, figsize=(16, 9))
    top.imshow(history.T, vmin=0, vmax=1, cmap="magma")
    bot.plot(history.sum(axis=1)[:-1], "b-", alpha=0.7, label="Num cells")
    bot.plot(diffs, "r-", alpha=0.7, label="Diffs")
    bot.grid()
    bot.legend()

    bot.set_xlim(0, history.shape[0])
    plt.tight_layout()
    plt.show()


def plot_evolution(gol: ThoroidalGOL, fps=10, repeats=10):
    for repeat in range(1, repeats+1):
        print(f"\nPlotting evolution, repeat {repeat}/{repeats}")
        for i, array in enumerate(gol.history, start=1):
            print(f"\rSimulation progress {i / len(gol.history):>7.2%}", end="")
            if i > 20:
                print(f" wmd: {gol.debug_logs['wmd'][i - 20]:.4f},"
                      f" wvd: {gol.debug_logs['wvd'][i - 20]:.4f}", end="")

            cv2.imshow("Array", array*255)
            if cv2.waitKey(1000 // fps) == 27:
                break
    print()
    cv2.destroyAllWindows()


def simulate():
    no_simulations = 10000
    convergence_types = defaultdict(int)
    for i in range(1, no_simulations+1):
        print(f"\rRunning simulation #{i}", end="")
        gol = ThoroidalGOL.from_random_states(alive_probability=0.4, convergence_memory=30)
        gol.simulate(steps=150)
        if gol.convergence_type == convergence.NONE:
            plot_history(gol)
            plot_evolution(gol, fps=10, repeats=1)
        if gol.convergence_type == convergence.CONSTANT:
            convergence_types[gol.convergence_constant] += 1
        else:
            convergence_types[gol.convergence_type] += 1

    pprint(convergence_types)
    sorted_keys = sorted(convergence_types.keys(), key=lambda k: f"{k:0>2}")
    barplot_x = [f"{k:0>2}" for k in sorted_keys]
    barplot_h = [convergence_types[t] for t in sorted_keys]
    plt.bar(barplot_x, barplot_h)
    plt.grid()
    plt.show()


if __name__ == '__main__':
    simulate()
