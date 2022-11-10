# Simulation 3D force relaxation in a group of spheroid cells

from vedo import ProgressBar, Spheres
import numpy as np
from Forces.forces import relu_force
from Solvers.solvers import take_euler_step

# Params
N = 200
T = 100
r_max = 1.0
r_eq = 0.8
dt = 0.1

# Initialise cells
ini_x = np.random.uniform(low=-1.0, high=1.0, size=N)
ini_y = np.random.uniform(low=-1.0, high=1.0, size=N)
ini_z = np.random.uniform(low=-1.0, high=1.0, size=N)

X = np.column_stack((ini_x, ini_y, ini_z))


pb = ProgressBar(0, T, c='red')
for t in pb.range():

    take_euler_step(X, N, dt, relu_force, r_max, r_eq)

    Spheres(X, c='b', r=0.3).show(axes=1, interactive=t <= 0)
    pb.print()
