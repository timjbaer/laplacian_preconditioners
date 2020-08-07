import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

import multigrid
#from agg import agg_multigrid

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
#        elif accel == "agg":
#            z,_,_ = agg_multigrid(A, r, u, 1, 2**2, N, 1) # Mz = r
        else:
            z = multigrid.v_cycle(A,u,r,1,2**2,numlvls,1) # Mz = r
        f = np.inner(r, z) / np.inner(r_prev, z_prev)
        p = z + f * p

    return x

#from test import mat_2d
#
#if __name__ == "__main__":
#    N = 64  # number of grid points (if fix boundary conditions, u_0=u_n=0, then N-2 DOFs)
#    n = N-1 # number of grid spaces
#    nf = n-1
#    nc = (n+1)//2 - 1
#
#    # setup problem system, Au=f
#    a = 0; b = 1
#    A = mat_2d(N,a,b)         # Poisson problem
#    x = np.random.rand(nf**2) # Initial guess
#    f = np.ones(nf**2)        # solution vector
#
#    # direct solve
#    u = sla.spsolve(A,f)
#    r = A@u - f
#
#    numlvls = 2
#    u_v3_pcg_mg = pcg(A, x, f, N, accel="mg")
#    print("PCG with Multigrid Gauss-Seidel error: " + str(la.norm(u_v3_pcg_mg - u)/la.norm(u)))
#
#    u_v3_pcg_v = pcg(A, x, f, N, accel="v_cycle", numlvls=numlvls)
#    print("PCG with V cycle Gauss-Seidel error: " + str(la.norm(u_v3_pcg_v - u)/la.norm(u)))
#
#    #u_v3_pcg_agg = pcg(A, x, f, N, accel="agg", numlvls=numlvls)
#    #print("PCG with V cycle Gauss-Seidel error: " + str(la.norm(u_v3_pcg_v - u)/la.norm(u)))
