import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import matplotlib.pyplot as plt

from multigrid import v_cycle, multigrid_v3

# form 2d poisson problem
def mat_2d(num_grid_pts, left_coor=0, right_coor=1):
    """ Sets up matrix for 2d Poisson problem
    @returns matrix: system for 2D points over equaspaced points [left,right]
    """
    ## 2D matrix setup
    x = np.linspace(left_coor,right_coor, num_grid_pts)
    h = x[1]-x[0] # length between two grid points
    A = sp.diags([1, -2, 1], [-1, 0, 1], shape=(N-2, N-2), format='csr')
    I = sp.eye(N-2,format='csr')
    return (1./h**2) * (sp.kron(I,A) + sp.kron(A,I))

def pcg(A, x, f, N, accel="mg", numlvls=2):
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
        z,_,_ = multigrid_v3(A, r, u, 1, 2**2, N, 1) # Mz = r
    else:
        z = v_cycle(A,u,r,1,2**2,numlvls,1) # Mz = r
    p = z

    for j in range(5):
        r_prev = r
        z_prev = z

        a = np.inner(r, z) / np.inner(A @ p, p)
        x = x + a * p
        r = r - a * A @ p
        u = np.zeros(x.size, dtype=float)
        if accel == "mg":
            z,_,_ = multigrid_v3(A, r, u, 1, 2**2, N, 1) # Mz = r
        else:
            z = v_cycle(A,u,r,1,2**2,numlvls,1) # Mz = r
        f = np.inner(r, z) / np.inner(r_prev, z_prev)
        p = z + f * p

    return x

if __name__ == "__main__":
    N = 32  # number of grid points (if fix boundary conditions, u_0=u_n=0, then N-2 DOFs)
    n = N-1 # number of grid spaces
    nf = n-1
    nc = (n+1)//2 - 1

    # setup problem system, Au=f
    a = 0; b = 1
    A = mat_2d(N,a,b)         # Poisson problem
    x = np.random.rand(nf**2) # Initial guess
    f = np.ones(nf**2)        # solution vector

    # direct solve
    u = sla.spsolve(A,f)
    r = A@u - f

    numlvls = 2
    u_v3_pcg_mg = pcg(A, x, f, N, accel="mg")
    print("PCG with Multigrid Gauss-Seidel error: " + str(la.norm(u_v3_pcg_mg - u)/la.norm(u)))

    u_v3_pcg_v = pcg(A, x, f, N, accel="v_cycle", numlvls=numlvls)
    print("PCG with V cycle Gauss-Seidel error: " + str(la.norm(u_v3_pcg_v - u)/la.norm(u)))
