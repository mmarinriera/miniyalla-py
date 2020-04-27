# Implementation of a cell monolayer undergoing bending due to a change in preferential curvature
# Adapted from https://github.com/germannp/yalla (see examples/apical_constriction.cu)

import numpy as np
from Forces.forces import apical_constriction_force
from Solvers.solvers import take_euler_step
from math import pi, sin, cos

# Transform polarity vectors from spherical to cartesian coordinates for visualisation
def arr_pol_to_float3(arr):
    out = np.zeros((arr.shape[0],3))
    for i in range(arr.shape[0]):
        out[i,0] = sin(arr[i,0]) * cos(arr[i,1])
        out[i,1] = sin(arr[i,0]) * sin(arr[i,1])
        out[i,2] = cos(arr[i,0])

    return out

# Params
N=100
T = 1000
output_int = int(T/100)
r_max = 1.0
r_eq = 0.8
dt = 0.1
preferential_angle = 60.0 * pi/180 # in radiants


# Initialise cells
ini_x = np.random.uniform(low=-1.0, high=1.0, size=N)
ini_y = np.random.uniform(low=-1.0, high=1.0, size=N)
ini_z = np.zeros(N)
ini_theta = np.zeros(N)
ini_phi = np.ones(N)

X = np.column_stack((ini_x, ini_y, ini_z, ini_theta, ini_phi))

# List to store time series output
coords_t=[]
pol_t=[]

# Save state at t=0
out_X = np.copy(X)
pol = arr_pol_to_float3(out_X[:,3:5])
coords_t.append(out_X[:,:3])
pol_t.append(pol)

from vtkplotter import *
import time
pb = ProgressBar(0, T, c='red')
for t in pb.range():
    take_euler_step(X, N, dt, apical_constriction_force, r_max, r_eq, preferential_angle)
    pb.print("Integrating")

    if t % output_int == 0:
        out_X = np.copy(X)
        pol = arr_pol_to_float3(out_X[:,3:5])
        coords_t.append(out_X[:,:3])
        pol_t.append(pol)


vp = Plotter(verbose=0, interactive=0)

vp.camera.SetPosition([20, 20, 10])
vp.camera.SetFocalPoint([2.5, 2.5, 0])
vp.camera.SetViewUp([0,0,1])

for t in range(len(coords_t)):

    vp.actors = []

    cells = Spheres(coords_t[t], c='b', r=0.4)
    polarities = Arrows(startPoints=coords_t[t], endPoints=coords_t[t] + pol_t[t], c='r')
    vp.add([cells, polarities])
    vp.show(resetcam=0)

vp.show(resetcam=0, interactive=1)
