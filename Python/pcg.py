import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

import multigrid, agg

def pcg(A, x, f, N, cgiter=5, accel="mg", numlvls=2):
    """
    @param A: matrix in system Ax=f
    @param x: initial guess to system Ax=f
    @param f: vector in system Ax=f
    @param numlvls: number of cycles in V cycle
    @return x: Approximate solution to Ax=f

    See Saad's book pages 276 - 281 for reference
    """
    r = f - A @ x
    u = np.zeros(x.size, dtype=float)
    if accel == "mg":
        z,_,_ = multigrid.v_cycle_3lvls(A, r, u, 1, 2**2, N, 1) # Mz = r
    else:
        z = multigrid.v_cycle(A,u,r,1,2**2,numlvls,1) # Mz = r
    p = z

    for j in range(cgiter):
        r_prev = r
        z_prev = z

        a = np.inner(r, z) / np.inner(A @ p, p)
        x = x + a * p
        r = r - a * A @ p
        u = np.zeros(x.size, dtype=float)
        if accel == "mg":
            z,_,_ = multigrid.v_cycle_3lvls(A, r, u, 1, 2**2, N, 1) # Mz = r
        elif accel == "agg":
            z,_,_ = agg.agg_multigrid(A, r, u, 1, 2**2, N, 1) # Mz = r
        else:
            z = multigrid.v_cycle(A,u,r,1,2**2,numlvls,1) # Mz = r
        f = np.inner(r, z) / np.inner(r_prev, z_prev)
        p = z + f * p

    return x
