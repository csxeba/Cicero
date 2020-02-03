import numpy as np
from scipy import stats
from matplotlib import pyplot as plt


def recenter(state):
    coords = np.argwhere(state > 0)
    circular_mean = np.floor(stats.circmean(coords, high=6, low=0, axis=0)).astype(int)
    centered_coords_raw = coords - circular_mean
    centered_coords = centered_coords_raw.copy()
    centered_coords[centered_coords < 0] = 6 + centered_coords[centered_coords < 0]
    centered_coords[centered_coords >= 6] = 6 - centered_coords[centered_coords >= 6]
    recentered_coords = (centered_coords + 2) % 6

    new_state = np.zeros_like(state)
    new_state[tuple(recentered_coords[:, 0]), tuple(recentered_coords[:, 1])] = 1

    torque = stats.circvar(recentered_coords, axis=0).sum()

    return new_state, torque


data = np.load("attractors.npy")

num_cells = data.sum(axis=(1, 2))

nonzero = np.nonzero(num_cells)
data = data[nonzero]
num_cells = num_cells[nonzero]

sorting_arg = np.argsort(num_cells)
sorted_data = data[sorting_arg]

uniques = None
hits = []
torques = None

for x_state in sorted_data:
    recentered, torque = recenter(x_state)
    recentered = recentered[None, ...]
    torque = np.array([torque])
    if uniques is None:
        uniques = recentered
        torques = torque
        hits.append(1)
        continue
    diffs = np.abs(torque - torques)
    if np.any(np.isclose(diffs, 0)):
        hits[np.argmin(diffs)] += 1
        continue
    uniques = np.concatenate([uniques, recentered], axis=0)
    torques = np.concatenate([torques, torque], axis=0)
    hits.append(1)

torques = np.array(torques)
hits = np.array(hits)
total_hits = sum(hits)
print("Found", len(uniques), "uniques")

num_cells = np.sum(uniques, axis=(1, 2))
sort_args = np.argsort(num_cells)
uniques = uniques[sort_args]
torques = torques[sort_args]
hits = hits[sort_args]
num_cells = num_cells[sort_args]

max_hit = int(np.argmax(hits))
plt.imshow(uniques[max_hit])
plt.title(f"Most likely attractor, p = {hits[max_hit] / total_hits:.2%}")
plt.show()

for unique, torque, hits in zip(uniques, torques, hits):
    plt.imshow(unique)
    plt.title(f"torque: {torque:.4f} hits: {hits/total_hits:.2%}")
    plt.show()
