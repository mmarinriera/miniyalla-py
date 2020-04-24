from numba import njit, prange

@njit(parallel=True)
def take_euler_step(X, dt, force_func, *args):

    # Compute differential displacements
    dX = force_func(X, *args)

    # Loop over all cells to update positions and polarities
    for i in prange(X.shape[0]):
        X[i,:] += dX[i,:] * dt
        # X[i,0] += dX[i,0] * dt
        # X[i,1] += dX[i,1] * dt
        # X[i,2] += dX[i,2] * dt
        # X[i,3] += dX[i,3] * dt
        # X[i,4] += dX[i,4] * dt
