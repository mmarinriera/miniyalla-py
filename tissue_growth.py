# Simulation of proliferating spheroid cells

from vedo import ProgressBar, Spheres, show
import numpy as np
from Forces.forces import relu_force
from Solvers.solvers import take_euler_step
from math import sin, cos, acos, pi
from numba import njit, prange

# Params
N = 10
n_max = 1000
T = 500
prolif_rate = 0.005
r_max = 1.0
r_eq = 0.8
dt = 0.1


@njit(parallel=True)
def proliferation(X, N, prolif_rate, mean_dist):
    marked_for_division = np.zeros(N)
    for i in prange(N):
        if np.random.uniform(0, 1) < prolif_rate:
            # Cell division happens
            marked_for_division[i] = 1

    marked_for_div_idx = np.where(marked_for_division == 1)[0]
    # Initialise new cells
    for it in prange(marked_for_div_idx.size):
        idx = marked_for_div_idx[it]
        idx_new = N + it

        theta = acos(2. * np.random.uniform(0, 1) - 1)
        phi = np.random.uniform(0, 1) * 2.0 * pi
        X[idx_new, 0] = X[idx, 0] + mean_dist / 4 * sin(theta) * cos(phi)
        X[idx_new, 1] = X[idx, 1] + mean_dist / 4 * sin(theta) * sin(phi)
        X[idx_new, 2] = X[idx, 2] + mean_dist / 4 * cos(theta)

    return N + marked_for_div_idx.size


# Initialise cells
ini_x = np.random.uniform(low=-1.0, high=1.0, size=n_max)
ini_y = np.random.uniform(low=-1.0, high=1.0, size=n_max)
ini_z = np.random.uniform(low=-1.0, high=1.0, size=n_max)

X = np.column_stack((ini_x, ini_y, ini_z))

########################################################

pb = ProgressBar(0, T, c='red')
for t in pb.range():

    take_euler_step(X, N, dt, relu_force, r_max, r_eq)
    N = proliferation(X, N, prolif_rate, r_eq)

    cells = Spheres(X[:N, :], c='b', r=0.4)
    show(cells, interactive=t <= 0, viewup='z')
    pb.print("N= " + str(N))
