import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import matplotlib.pyplot as plt

# Jacobi Smoother
def wtd_jacobi(A,x,b,w,numiter):
    """
    @param A: matrix in system Ax=b
    @param x: initial guess to system Au=f
    @param b: vector in system Au=f
    @param w: weight/dampening constant
    @param numiter: number of iterations
    @return u: Approximate solution Au=b
    """
    D = A.diagonal()
    R = A.copy()
    R.setdiag(0)
    u = x.copy()
    for _ in range(numiter):
        u = w * (D**(-1) * (b-R@u)) + (1-w) * u
    return u

# Gauss-Seidel Smoother
def gauss_seidel(A,x,b,w,numiter):
    """
    @param A: matrix in system Ax=b
    @param x: initial guess to system Au=f
    @param b: vector in system Au=f
    @param w: weight/dampening constant
    @param numiter: number of iterations
    @return u: Approximate solution Au=b
    """
    L = sp.tril(A,0).tocsr()
    U = sp.triu(A,1).tocsr()
    u = x.copy()
    for _ in range(numiter):
        u = sla.spsolve(L,b-U@u)
    return u

# 1d interpolation operator
def interpolation_1d(nc, nf):
    """Simple interpolator from ceil(n/2)-1 => n-1 
    of nearby grid points
    @param nc: number of course grid points
    @param nf: number of fine grids. @nf not used since n is well defined with @nc
    """
    d = np.repeat([[1, 2, 1]], nc, axis=0).T
    I = np.zeros((3,nc), dtype=int)
    for i in range(nc):
        I[:,i] = [2*i, 2*i+1, 2*i+2]
    J = np.repeat([np.arange(nc)], 3, axis=0)
    P = sp.coo_matrix(
        (d.ravel(), (I.ravel(), J.ravel()))
        ).tocsr()
    return 0.5 * P

def restrictor_1d(nc, nf):
    return interpolation_1d(nc,nf).T

def v_cycle(A,x,f,w,numiter,numlvls=1,ind=0,exact_solve=False):
    """
    @param A: matrix in system Ax=f
    @param x: initial guess to system Ax=f
    @param f: vector in system Ax=f
    @param w: weight/dampening constant
    @param numiter: number of iterations for relaxation
    @param numlvls: number of cycles in V cycle
    @param ind: interger switcher for smoother
    @return u: Approximate solution Au=b
    """
    # define problem sizes
    nf = int(A.shape[0]**.5)     # number of grids along an axis @ curr lvl
    assert(nf == A.shape[0]**.5) # ensure is square
    n  = nf+1
    nc = (n+1)//2 - 1            # number of grid points in coarse

    # defining interpolating and restrictor operator
    I = interpolation_1d(nc, nf)[:nf,:nc]  # edge-case handling
    I_2d = sp.kron(I,I).tocsr()
    J = restrictor_1d(nc,nf)[:nc,:nf]
    J_2d = sp.kron(J,J).tocsr()

    # relax on current grid and calculate residual
    smooth = gauss_seidel if ind==1 else wtd_jacobi
    u0 = x.copy()
    u1 = smooth(A,u0,f,w,numiter)
    r1_f = f-A@u1

    # restrict both residual and system to coarser grid 
    A_c  = J_2d @ A @ J_2d.T
    r1_c = J_2d @ r1_f

    # recurse
    if(numlvls > 1):
        e = np.zeros(nc**2) # initial guess
        u2 = v_cycle(A_c,e,r1_c,w,numiter,numlvls-1,ind,exact_solve)
    else:
        if(exact_solve):
            u2 = sla.spsolve(A_c,r1_c)
        else:
            u2 = np.zeros(nc**2)

    # interpolate residual solution and add to current approximation
    u = u1 + I_2d@u2

    # Final smoothing: use @u as initial guess
    u = smooth(A,u,f,w,numiter)

    return u

# multigrid 3 level
def multigrid_v3(A,f,x,w,numiter,N,ind):
    # form multigrid operators for all levels
    nf = int(A.shape[0]**.5)     # number of grids along an axis @ curr lvl
    assert(nf == A.shape[0]**.5) # ensure is square
    n  = nf+1
    nc1 = (n+1)//2 - 1            # number of grid points in coarse lvl 1
    nc2 = (nc1+1)//2 - 1          # number of grid points in coarse lvl 2

    # N1 = int( (N-1)/2 + 1)
    # N2 = int( (N1-1)/2 + 1)
    # P1 = interpolation_1d(N1-2,N-2)
    # P2 = interpolation_1d(N2-2,N1-2)

    P1 = interpolation_1d(nf,nc1)[:nf,:nc1]     # subset for off-by-one error
    P2 = interpolation_1d(nc1,nc2)[:nc1,:nc2]
    P1_2D = sp.kron(P1,P1).tocsr()
    P2_2D = sp.kron(P2,P2).tocsr()
    A1 = P1_2D.T@  A @ P1_2D
    A2 = P2_2D.T@ A1 @P2_2D

    ## level 0 ##
    smooth = gauss_seidel if ind==1 else wtd_jacobi
    u = smooth(A,x,f,w,numiter) # pre-smoothing
    r = f - A@u # form residual
    f1 = P1_2D.T@r # restrict

    ## Level 1 ##
    e = np.zeros(nc1**2)
    u1 = smooth(A1,e,f1,w,numiter) # smoothing
    r1 = f1 - A1@u1 # form residual
    f2 = P2_2D.T@r1 # restrict

    ## Level 2 ##
    u2 = sla.spsolve(A2,f2) #solve system

    ## Level 1 ##
    u1 = u1 + P2_2D@u2 # interpolate and correct previous solution
    u1 = smooth(A1,u1,f1,w,numiter) # smoothing

    ## Level 0 ##
    u = u + P1_2D@u1 # interpolate and correct previous solution

    u = smooth(A,u,f,w,numiter) # post-smoothing
    return u, A1, A2

#from test import mat_2d
#
#if __name__ == "__main__":
#    N = 32  # number of grid points (if fix boundary conditions, u_0=u_n=0, then N-2 DOFs)
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
#    # multigrid acceleration
#    w = 2./3
#    iter_array = [2**k for k in range(7)]
#
#    err_v3_j = []
#    err_v3_wj = []
#    err_v3_gs = []
#
#    show_plot = False
#    numlvls=2
#    if not show_plot:
#        u_v3_j,_,_ = multigrid_v3(A,x,f,1,iter_array[-1],N,0)
#        u_v3_wj,_,_ = multigrid_v3(A,x,f,w,iter_array[-1],N,0)
#        u_v3_gs,_,_ = multigrid_v3(A,x,f,1,iter_array[-1],N,1)
#
#        print("V cycle Jacobi error: " + str(la.norm(u_v3_j - u)/la.norm(u)))
#        print("V cycle wtd Jacobi error: " + str(la.norm(u_v3_wj - u)/la.norm(u)))
#        print("V cycle Gauss-Seidel error: " + str(la.norm(u_v3_gs - u)/la.norm(u)))
#
#        print("\n=== Alt methods (w/o exact solve) ===")
#
#        u_v3_j = v_cycle(A,x,f,1,iter_array[-1],numlvls,0)
#        u_v3_wj = v_cycle(A,x,f,w,iter_array[-1],numlvls,0)
#        u_v3_gs = v_cycle(A,x,f,1,iter_array[-1],numlvls,1)
#
#        print("V cycle Jacobi error: " + str(la.norm(u_v3_j - u)/la.norm(u)))
#        print("V cycle wtd Jacobi error: " + str(la.norm(u_v3_wj - u)/la.norm(u)))
#        print("V cycle Gauss-Seidel error: " + str(la.norm(u_v3_gs - u)/la.norm(u)))
#
#        print("\n=== Alt methods (w exact solve) ===")
#
#        u_v3_j = v_cycle(A,x,f,1,iter_array[-1],numlvls,0,exact_solve=True)
#        u_v3_wj = v_cycle(A,x,f,w,iter_array[-1],numlvls,0,exact_solve=True)
#        u_v3_gs = v_cycle(A,x,f,1,iter_array[-1],numlvls,1,exact_solve=True)
#
#        print("V cycle Jacobi error: " + str(la.norm(u_v3_j - u)/la.norm(u)))
#        print("V cycle wtd Jacobi error: " + str(la.norm(u_v3_wj - u)/la.norm(u)))
#        print("V cycle Gauss-Seidel error: " + str(la.norm(u_v3_gs - u)/la.norm(u)))
#
#    else:
#        for numiter in iter_array:
#            u_v3_j,_,_ = multigrid_v3(A,x,f,1,numiter,N,0)
#            u_v3_wj,_,_ = multigrid_v3(A,x,f,w,numiter,N,0)
#            u_v3_gs,_,_ = multigrid_v3(A,x,f,1,numiter,N,1)
#
#            """ Recursive v-cycle code
#            u_v3_j = v_cycle(A,x,f,1,numiter,numlvls,0,exact_solve=True)
#            u_v3_wj = v_cycle(A,x,f,w,numiter,numlvls,0,exact_solve=True)
#            u_v3_gs = v_cycle(A,x,f,1,numiter,numlvls,1,exact_solve=True)
#            """
#
#            err_v3_j.append(la.norm(u_v3_j -u)/la.norm(u))
#            err_v3_wj.append(la.norm(u_v3_wj -u)/la.norm(u))
#            err_v3_gs.append(la.norm(u_v3_gs -u)/la.norm(u))
#
#        # plot the errors
#        plt.figure()
#        plt.semilogy(iter_array,err_v3_j,label = 'V cycle Jacobi', color = 'b', linestyle = '-')
#        plt.semilogy(iter_array,err_v3_wj,label = 'V cycle wtd Jacobi', color = 'g', linestyle = '-')
#        plt.semilogy(iter_array,err_v3_gs,label = 'V cycle Gauss-Seidel', color = 'r', linestyle = '-')
#        plt.title('Multigrid error vs iteration')
#        plt.ylabel('Error')
#        plt.xlabel('Number of smoother iterations')
#        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#        plt.show()
