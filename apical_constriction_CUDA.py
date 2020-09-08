# Implementation of a cell monolayer undergoing bending due to a change in preferential curvature
# Adapted from https://github.com/germannp/yalla (see examples/apical_constriction.cu)

import numpy as np
from numba import cuda
from math import cos, sin, acos, atan2, pow, pi

N=100
T = 1000
output_int = int(T/100)
r_max = 1.0
r_eq = 0.8
dt = 0.1
preferential_angle = 60.0 * pi/180 # in radiants

threadsperblock = 32
blockspergrid = (N + (threadsperblock - 1)) // threadsperblock

@cuda.jit(debug=True)
def compute_displacements(X, dX, n):

    #Determine element position in array using grid object
    i = cuda.grid(1)

    # Check that it's not outside the array
    if i < n :

        # Initialise variables
        Xix = X[i,0]
        Xiy = X[i,1]
        Xiz = X[i,2]
        Xitheta = X[i,3]
        Xiphi = X[i,4]

        dXx = 0.0
        dXy = 0.0
        dXz = 0.0
        dXtheta = 0.0
        dXphi = 0.0

        # Scan neighbours
        for j in range(n):
            if i != j :
                Xj = X[j,:]
                rx = Xix-Xj[0]
                ry = Xiy-Xj[1]
                rz = Xiz-Xj[2]

                dist = pow(rx*rx + ry*ry + rz*rz, 0.5)
                if dist < r_max:
                    # Calculate attraction/repulsion force differential here
                    F = max(r_eq - dist, 0)*2.0 - max(dist - r_eq, 0)*1.5
                    dXx += rx * F / dist
                    dXy += ry * F / dist
                    dXz += rz * F / dist

                    # Apical constriction force
                    # Adapted from https://github.com/germannp/yalla (see include/polarity.cuh)
                    dBx = 0.0
                    dBy = 0.0
                    dBz = 0.0
                    dBtheta = 0.0
                    dBphi = 0.0

                    # Polarity vector of i in cartesian coordinates
                    pix = sin(Xitheta) * cos(Xiphi)
                    piy = sin(Xitheta) * sin(Xiphi)
                    piz = cos(Xitheta)

                    # dot product between polarity of i and r vector plus correction for preferential angle
                    prodi = (pix * rx + piy * ry + piz * rz) / dist + cos(preferential_angle)

                    # transform r vector to spherical coordinates
                    r_hat_theta = acos(rz / dist)
                    r_hat_phi = atan2(ry, rx)

                    # Compute rotation to polarity vector
                    dBtheta = -prodi*(cos(Xitheta) * sin(r_hat_theta) * cos(Xiphi - r_hat_phi) - sin(Xitheta) * cos(r_hat_theta))
                    sin_Xi_theta = sin(Xitheta)
                    if abs(sin_Xi_theta) > 1e-10:
                        dBphi = -prodi*(-sin(r_hat_theta) * sin(Xiphi - r_hat_phi) / sin_Xi_theta)

                    # Compute displacement due to bending force
                    dBx = -prodi / dist * pix + prodi*prodi / (dist*dist) * rx
                    dBy = -prodi / dist * piy + prodi*prodi / (dist*dist) * ry
                    dBz = -prodi / dist * piz + prodi*prodi / (dist*dist) * rz

                    # Bending force contribution to Pj
                    Xjtheta = X[j,3]
                    Xjphi = X[j,4]

                    # Polarity vector of j in cartesian coordinates
                    pjx = sin(Xjtheta) * cos(Xjphi)
                    pjy = sin(Xjtheta) * sin(Xjphi)
                    pjz = cos(Xjtheta)

                    # Apply displacement to j due to bending force
                    prodj = (pjx * rx + pjy * ry + pjz * rz) / dist - cos(preferential_angle)
                    dBx += -prodj / dist * pjx + prodj*prodj / (dist*dist) * rx
                    dBy += -prodj / dist * pjy + prodj*prodj / (dist*dist) * ry
                    dBz += -prodj / dist * pjz + prodj*prodj / (dist*dist) * rz

                    # Add apical constriction force contribution to main displacement
                    factor = 0.2
                    dXx += factor * dBx
                    dXy += factor * dBy
                    dXz += factor * dBz
                    dXtheta += factor * dBtheta
                    dXphi += factor * dBphi

        # Update displacement array
        dX[i,0] = dXx
        dX[i,1] = dXy
        dX[i,2] = dXz
        dX[i,3] = dXtheta
        dX[i,4] = dXphi


@cuda.jit(debug=True)
def update_positions(X, dX, n):
    #Determine element position in array using grid object
    i = cuda.grid(1)

    # Check that it's not outside the array
    if i < n :
        # Update positions by Euler method
        X[i,0] += dX[i,0] * dt
        X[i,1] += dX[i,1] * dt
        X[i,2] += dX[i,2] * dt
        X[i,3] += dX[i,3] * dt
        X[i,4] += dX[i,4] * dt


def take_step(X, n):

    # Initialise displacement array
    dX = cuda.to_device(np.zeros(shape=(n,5)))

    compute_displacements[blockspergrid, threadsperblock](X, dX, n)
    update_positions[blockspergrid, threadsperblock](X, dX, n)

# Transform polarity vectors from spherical to cartesian coordinates for visualisation
def arr_pol_to_float3(arr):
    out = np.zeros((arr.shape[0],3))
    for i in range(arr.shape[0]):
        out[i,0] = sin(arr[i,0]) * cos(arr[i,1])
        out[i,1] = sin(arr[i,0]) * sin(arr[i,1])
        out[i,2] = cos(arr[i,0])

    return out

# Initialise cells
ini_x = np.random.uniform(low=0.0, high=2.00, size=N)
ini_y = np.random.uniform(low=0.0, high=2.00, size=N)
#ini_z = np.random.uniform(low=0.0, high=2.00, size=N)
ini_z = np.zeros(N)
ini_theta = np.zeros(N)
ini_phi = np.ones(N)

h_X = np.column_stack((ini_x, ini_y, ini_z, ini_theta, ini_phi))
d_X = cuda.to_device(h_X)

# List to store time series output
coords_t=[]
pol_t=[]

# Save state at t=0
out_X = d_X.copy_to_host()

pol = arr_pol_to_float3(out_X[:,3:5])
coords_t.append(out_X[:,:3])
pol_t.append(pol)


for t_interval in range(int(T/output_int)):
    for it in range(output_int):
        t=t_interval*output_int + it
        #print('\r', "t=", t, end='')
        #print("t=", t)
        take_step(d_X, N)

    print('\r', "t=", t, end='')
    out_X = d_X.copy_to_host()

    pol = arr_pol_to_float3(out_X[:,3:5])
    coords_t.append(out_X[:,:3])
    pol_t.append(pol)

#print('\r', "DONE", t, end='')

#######################################################################
from vedo import Box, Spheres, Arrows, show, interactive

world = Box(pos=(1,1,0), size=(10,10,4), alpha=1).wireframe()

for t in range(len(coords_t)):
    cells = Spheres(coords_t[t], c='b', r=0.4)
    polarities = Arrows(startPoints=coords_t[t],
                        endPoints=coords_t[t] + pol_t[t], c='tomato')
    show(world, cells, polarities, interactive=0, viewup='z')

interactive()

