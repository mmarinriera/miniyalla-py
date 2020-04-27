from numba import njit, prange

@njit(parallel=True)
def take_euler_step(X, dt, force_func, *args):

    # Compute differential displacements
    dX = force_func(X, *args)

    # Loop over all cells to update positions and polarities
    for i in prange(X.shape[0]):
        X[i,:] += dX[i,:] * dt
