# Simulation 3D force relaxation in a group of spheroid cells

import numpy as np
from Forces.forces import relu_force
from Solvers.solvers import take_euler_step

# Params
N=100
T = 100
output_int = int(T/100)
r_max = 1.0
r_eq = 0.8
dt = 0.01

# Initialise cells
ini_x = np.random.uniform(low=-1.0, high=1.0, size=N)
ini_y = np.random.uniform(low=-1.0, high=1.0, size=N)
ini_z = np.random.uniform(low=-1.0, high=1.0, size=N)

X = np.column_stack((ini_x, ini_y, ini_z))

from vtkplotter import *

vp = Plotter(verbose=0, interactive=0)
vp.camera.SetPosition([20, 20, 10])
vp.camera.SetFocalPoint([0, 0, 0])
vp.camera.SetViewUp([0,0,1])

# Display state at t=0
cells = Spheres(X, c='b', r=0.4)
vp.add(cells)
vp.show(resetcam=0)

pb = ProgressBar(0, T, c='red')
for t in pb.range():

    take_euler_step(X, dt, relu_force, r_max, r_eq)

    pb.print()
    vp.actors = []

    cells = Spheres(X, c='b', r=0.4)
    vp.add(cells)
    vp.show(resetcam=0)


vp.show(resetcam=0, interactive=1)
