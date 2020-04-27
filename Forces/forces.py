import numpy as np
from numba import njit, prange
from math import cos, sin, acos, atan2, pow


@njit(parallel=True)
def relu_force(X, n, r_max, r_eq):

    # Initialise displacement array
    dX = np.zeros(shape=(n, X.shape[1]))

    # Loop over all cells to compute displacements
    for i in prange(n):

        # Initialise variables
        Xi = X[i,:]

        # Scan neighbours
        for j in range(n):
            if i != j :
                r = Xi - X[j,:]
                dist = np.linalg.norm(r)
                if dist < r_max:
                    # Calculate attraction/repulsion force differential here
                    F = max(r_eq - dist, 0)*2.0 - max(dist - r_eq, 0)*0.5
                    dX[i,:] += r * F /dist

    return dX


@njit(parallel=True)
def apical_constriction_force(X, n, r_max, r_eq, pref_angle):

    # Initialise displacement array
    dX = np.zeros(shape=(n, X.shape[1]))

    # Loop over all cells to compute displacements
    for i in prange(n):

        # Initialise variables
        # Xi = X[i,:]
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
                # r = Xi - X[j,:]
                rx = Xix-X[j,0]
                ry = Xiy-X[j,1]
                rz = Xiz-X[j,2]
                # dist = np.linalg.norm(r)
                dist = pow(rx*rx + ry*ry + rz*rz, 0.5)

                if dist < r_max:
                    # Calculate attraction/repulsion force differential here
                    F = max(r_eq - dist, 0) * 2.0 - max(dist - r_eq, 0) * 1.5
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
                    piz = cos(Xitheta);

                    # dot product between polarity of i and r vector plus correction for preferential angle
                    prodi = (pix * rx + piy * ry + piz * rz) / dist + cos(pref_angle)

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
                    pjz = cos(Xjtheta);

                    # Compute displacement to j due to bending force
                    prodj = (pjx * rx + pjy * ry + pjz * rz) / dist - cos(pref_angle)
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

    return dX
