import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import matplotlib.pyplot as plt

# form 2d poisson problem

def mat_2d(N,a,b):
    ## 2D matrix setup
    x = np.linspace(a,b,N)
    h = x[1]-x[0]
    ivh2 = 1./(h**2)
    A = sp.diags([1, -2, 1], [-1, 0, 1], shape=(N-2, N-2), format='csr')
    I = sp.eye(N-2,format='csr')
    A_2d = ivh2 * (sp.kron(I,A) + sp.kron(A,I))
    return A_2d

# Jacobi Smoother

def wtd_jacobi(A,b,x,w,iter):
    D = A.diagonal()
    R = A.copy()
    R.setdiag(0)
    u = x.copy()
    i = 1
    while i <= iter:
        u = w * (D**(-1) * (b-R@u)) + (1-w) * u
        i += 1
    return u

# Gauss-Seidel Smoother

def gauss_seidel(A,b,x,w,iter):
    L = sp.tril(A,0).tocsr()
    U = sp.triu(A,1).tocsr()
    u = x.copy()
    i = 1
    while i <= iter:
        u = sla.spsolve(L,b-U@u)
        i += 1
    return u

# 1d interpolation operator

def interpolation1d(nc, nf):
    d = np.repeat([[1, 2, 1]], nc, axis=0).T
    I = np.zeros((3,nc), dtype=int)
    for i in range(nc):
        I[:,i] = [2*i, 2*i+1, 2*i+2]
    J = np.repeat([np.arange(nc)], 3, axis=0)
    P = sp.coo_matrix(
        (d.ravel(), (I.ravel(), J.ravel()))
        ).tocsr()
    return 0.5 * P

# multigrid 3 level

def multigrid_v3(A,f,x,w,iter,N,ind):

    # form multigrid operators for all levels
    N1 = int( (N-1)/2 + 1)
    N2 = int( (N1-1)/2 + 1)
    P1 = interpolation1d(N1-2,N-2)
    P2 = interpolation1d(N2-2,N1-2)
    P1_2D = sp.kron(P1,P1).tocsr()
    P2_2D = sp.kron(P2,P2).tocsr()
    A1 = P1_2D.T@A@ P1_2D
    A2 = P2_2D.T@A1@P2_2D

    smooth = wtd_jacobi

    if ind == 1:
        smooth = gauss_seidel

    ## level 0 ##
    u = smooth(A,f,x,w,iter) # pre-smoothing
    r = f - A@u # form residual
    f1 = P1_2D.T@r # restrict

    ## Level 1 ##
    e = np.zeros((N1-2)**2)
    u1 = smooth(A1,f1,e,w,iter) # smoothing
    r1 = f1 - A1@u1 # form residual
    f2 = P2_2D.T@r1 # restrict

    ## Level 2 ##
    u2 = sla.spsolve(A2,f2) #solve system

    ## Level 1 ##
    u1 = u1 + P2_2D@u2 # interpolate and correct previous solution
    u1 = smooth(A1,f1,u1,w,iter) # smoothing

    ## Level 0 ##
    u = u + P1_2D@u1 # interpolate and correct previous solution

    u = smooth(A,f,u,w,iter) # post-smoothing
    return u, A1, A2

if __name__ == "__main__":
    N = 33
    x = np.random.rand((N-2) **2)

    a = 0
    b = 1

    A = mat_2d(N,a,b)
    f = np.ones((N-2) **2)

    # direct solve

    u = sla.spsolve(A,f)
    r = A@u - f

    # multigrid acceleration

    w = 2/3
    iter_array = [2**k for k in range(7)]

    err_v3_j = []
    err_v3_wj = []
    err_v3_gs = []

    show_plot = False
    if not show_plot:
        u_v3_j,_,_ = multigrid_v3(A,f,x,1,iter_array[-1],N,0)
        u_v3_wj,_,_ = multigrid_v3(A,f,x,w,iter_array[-1],N,0)
        u_v3_gs,_,_ = multigrid_v3(A,f,x,1,iter_array[-1],N,1)

        print("V cycle Jacobi error: " + str(la.norm(u_v3_j - u)/la.norm(u)))
        print("V cycle wtd Jacobi error: " + str(la.norm(u_v3_wj - u)/la.norm(u)))
        print("V cycle Gauss-Seidel error: " + str(la.norm(u_v3_gs - u)/la.norm(u)))

    else:
        for iter in iter_array:
            u_v3_j,_,_ = multigrid_v3(A,f,x,1,iter,N,0)
            u_v3_wj,_,_ = multigrid_v3(A,f,x,w,iter,N,0)
            u_v3_gs,_,_ = multigrid_v3(A,f,x,1,iter,N,1)

            err_v3_j.append(la.norm(u_v3_j -u)/la.norm(u))
            err_v3_wj.append(la.norm(u_v3_wj -u)/la.norm(u))
            err_v3_gs.append(la.norm(u_v3_gs -u)/la.norm(u))

        # plot the errors
        plt.figure()
        plt.loglog(iter_array,err_v3_j,label = 'V cycle Jacobi', color = 'b', linestyle = '-')
        plt.loglog(iter_array,err_v3_wj,label = 'V cycle wtd Jacobi', color = 'g', linestyle = '-')
        plt.loglog(iter_array,err_v3_gs,label = 'V cycle Gauss-Seidel', color = 'r', linestyle = '-')
        plt.title('Multigrid error vs iteration')
        plt.ylabel('Error')
        plt.xlabel('Number of smoother iterations')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()
