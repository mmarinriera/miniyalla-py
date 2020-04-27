from numba import njit, prange

@njit(parallel=True)
def take_euler_step(X, N, dt, force_func, *args):

    # Compute differential displacements
    dX = force_func(X, N, *args)

    # Loop over all cells to update positions and polarities
    for i in prange(N):
        X[i,:] += dX[i,:] * dt
