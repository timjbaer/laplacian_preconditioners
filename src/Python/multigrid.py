import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import coarseners, relaxations

def v_cycle_with_fun(model_fun, smoother, u, f, w, numiter, numlvl=1):
    """ Runs v-cycle with a function that generates model problem

    Paramters
    ----------
    model_fun : function
        function that generates the model problem
    smoother : function
        function that relaxes the problem
    u : np.ndarray
        initial guess
    f : np.ndarray
        solution vector to system Au=f
    w : float
        damping factor 
    numiter : int
        number of iters to run
    numlvls : int
        number of v-cycles
    """
    nf = len(u)
    A = model_fun(nf)
    
    # relax
    v,_ = smoother(A,u,f,w,maxiter=numiter)
    
    # comptue residual and restrict
    r_f = f-A@v
    I = oned_injector(nf) 
    r_c = I@r_f
    
    # recurse onto residual if sufficient size, else don't recurse
    nc = I.shape[0]
    if(numlvl > 1):
        y = vcycle(model_fun, smoother, np.zeros(nc), r_c, w, numiter, numlvl-1)
    else:
        y = np.zeros(nc)
    
    # interpolate solution up and add
    J = oned_prolong(nc)[:nf]  
    z = v + J@y
    
    return z

def u_cycle_with_fun(mu, model_fun, smoother, u, f, w, numiter, numlvl=1):
    nf = len(u)
    A = model_fun(nf)
    
    # relax
    v,_ = smoother(A,u,f,w,maxiter=numiter)
    
    # comptue residual and restrict
    r_f = f-A@v
    I = oned_injector(nf) 
    r_c = I@r_f
    
    # recurse onto residual if sufficient size, else don't recurse
    nc = I.shape[0]
    y = np.zeros(nc)
    if(numlvl > 1):
        for _ in range(mu):
            y = ucycle(mu, model_fun, smoother, y, r_c, w, numiter, numlvl-1)
    else:
        pass
    
    # interpolate solution up and add
    J = oned_prolong(nc)[:nf]  
    z = v + J@y
    
    return z

def fmg_with_fun(model_fun, smoother, u, f, w, numiter, numlvl=1):
    # recurse onto residual if sufficient size, else don't recurse
    nf = len(u); 
    if(numlvl > 1):
        I = oned_injector(nf) 
        nc = I.shape[0]
        f_c = I@f_c
        v = fmg(model_fun, smoother, np.zeros(nc), f_c, w, numiter, numlvl-1)
    else:
        v = np.zeros(nc)
        
    # interpolate
    J = oned_prolong(nc)[:nf]  
    v0 = J@v
    
    # relax
    A = model_fun(nf)
    z,_ = smoother(A,v0,f,w,maxiter=maxiter)
    
    return z

def v_cycle(A,x,f,w,numiter,numlvls=1,ind=0,exact_solve=True):
    """

    Parameters
    ----------
    A,x,f: np.ndarrays
        matrix in system Ax=f
    w : float
        weight/dampening constant
    numiter : int
        number of iterations for relaxation
    numlvls : int
        number of cycles in V cycle
    ind : boolean
        integer switcher for smoother

    Returns
    ----------
    u : np.ndarray
        Approximate solution Au=b
    """
    # define problem sizes
    nf = int(A.shape[0]**.5)     # number of grids along an axis @ curr lvl
    assert(nf == A.shape[0]**.5) # ensure is square
    n  = nf+1
    nc = (n+1)//2 - 1            # number of grid points in coarse

    # defining interpolating and restrictor operator
    I = coarseners.interpolation_1d(nc, nf)[:nf,:nc]  # edge-case handling
    I_2d = sp.kron(I,I).tocsr()
    J = coarseners.restrictor_1d(nc,nf)[:nc,:nf]
    J_2d = sp.kron(J,J).tocsr()

    # relax on current grid and calculate residual
    smooth = relaxations.gauss_seidel if ind==1 else relaxations.wtd_jacobi
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
            u2 = spla.spsolve(A_c,r1_c)
        else:
            u2 = np.zeros(nc**2)

    # interpolate residual solution and add to current approximation
    u = u1 + I_2d@u2

    # Final smoothing: use @u as initial guess
    u = smooth(A,u,f,w,numiter)

    return u

# multigrid 3 level
def v_cycle_3lvls(A,f,x,w,numiter,N,ind):
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

    P1 = coarseners.interpolation_1d(nf,nc1)[:nf,:nc1]     # subset for off-by-one error
    P2 = coarseners.interpolation_1d(nc1,nc2)[:nc1,:nc2]
    P1_2D = sp.kron(P1,P1).tocsr()
    P2_2D = sp.kron(P2,P2).tocsr()
    A1 = P1_2D.T@  A @ P1_2D
    A2 = P2_2D.T@ A1 @P2_2D

    ## level 0 ##
    smooth = relaxations.gauss_seidel if ind==1 else relaxations.wtd_jacobi
    u = smooth(A,x,f,w,numiter) # pre-smoothing
    r = f - A@u # form residual
    f1 = P1_2D.T@r # restrict

    ## Level 1 ##
    e = np.zeros(nc1**2)
    u1 = smooth(A1,e,f1,w,numiter) # smoothing
    r1 = f1 - A1@u1 # form residual
    f2 = P2_2D.T@r1 # restrict

    ## Level 2 ##
    u2 = spla.spsolve(A2,f2) #solve system

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
