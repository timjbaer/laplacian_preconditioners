import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import matplotlib.pyplot as plt

import argparse

from multigrid import v_cycle, multigrid_v3
from pcg import pcg
#from agg import agg_multigrid

# form 2d poisson problem
def mat_2d(num_grid_pts, left_coor=0, right_coor=1):
    """ Sets up matrix for 2d Poisson problem
    @returns matrix: system for 2D points over equaspaced points [left,right]
    """
    ## 2D matrix setup
    x = np.linspace(left_coor,right_coor, num_grid_pts)
    h = x[1]-x[0] # length between two grid points
    #A = sp.diags([1, -2, 1], [-1, 0, 1], shape=(N-2, N-2), format='csr')
    A = sp.diags([1, -2, 1], [-1, 0, 1], shape=(num_grid_pts-2, num_grid_pts-2), format='csr')
    #I = sp.eye(N-2,format='csr')
    I = sp.eye(num_grid_pts-2,format='csr')
    return (1./h**2) * (sp.kron(I,A) + sp.kron(A,I))

if __name__ == "__main__":
    N = 32  # number of grid points (if fix boundary conditions, u_0=u_n=0, then N-2 DOFs)
    PLOT = False

    parser = argparse.ArgumentParser(description="""
                                        This script runs AMG on the laplacian of a specified graph
                                        """)
    parser.add_argument("-n", help="number of grid points")
    parser.add_argument("-plot", help="1 to show plot")

    args = parser.parse_args()

    if args.n != None: N = int(args.n)
    if args.plot != None: 
        if int(args.plot) == 1:
            PLOT = True 

    print("number of grid points N: " + str(N))

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

    # multigrid acceleration
    w = 2./3
    iter_array = [2**k for k in range(7)]

    numlvls=2
    cgiter = 3

    if not PLOT:
        u_v3_j,_,_ = multigrid_v3(A,x,f,1,iter_array[-1],N,0)
        u_v3_wj,_,_ = multigrid_v3(A,x,f,w,iter_array[-1],N,0)
        u_v3_gs,_,_ = multigrid_v3(A,x,f,1,iter_array[-1],N,1)

        print("\n=== Two level multigrid (w exact solve) ===")

        print("Two level multigrid Jacobi error: " + str(la.norm(u_v3_j - u)/la.norm(u)))
        print("Two level multigrid wtd Jacobi error: " + str(la.norm(u_v3_wj - u)/la.norm(u)))
        print("Two level multigrid Gauss-Seidel error: " + str(la.norm(u_v3_gs - u)/la.norm(u)))

        print("\n=== V-cycle (w/o exact solve) ===")

        u_v3_j = v_cycle(A,x,f,1,iter_array[-1],numlvls,0)
        u_v3_wj = v_cycle(A,x,f,w,iter_array[-1],numlvls,0)
        u_v3_gs = v_cycle(A,x,f,1,iter_array[-1],numlvls,1)

        print("V cycle Jacobi error: " + str(la.norm(u_v3_j - u)/la.norm(u)))
        print("V cycle wtd Jacobi error: " + str(la.norm(u_v3_wj - u)/la.norm(u)))
        print("V cycle Gauss-Seidel error: " + str(la.norm(u_v3_gs - u)/la.norm(u)))

        print("\n=== V-cycle (w exact solve) ===")

        u_v3_j = v_cycle(A,x,f,1,iter_array[-1],numlvls,0,exact_solve=True)
        u_v3_wj = v_cycle(A,x,f,w,iter_array[-1],numlvls,0,exact_solve=True)
        u_v3_gs = v_cycle(A,x,f,1,iter_array[-1],numlvls,1,exact_solve=True)

        print("V cycle Jacobi error: " + str(la.norm(u_v3_j - u)/la.norm(u)))
        print("V cycle wtd Jacobi error: " + str(la.norm(u_v3_wj - u)/la.norm(u)))
        print("V cycle Gauss-Seidel error: " + str(la.norm(u_v3_gs - u)/la.norm(u)))

        print("\n=== Preconitioned Conjugate Gradient ===")

        u_v3_pcg_mg = pcg(A,x,f,N,cgiter=cgiter,accel="mg")
        print("PCG with Multigrid Gauss-Seidel error (w exact solve): " + str(la.norm(u_v3_pcg_mg - u)/la.norm(u)))

        u_v3_pcg_v = pcg(A,x,f,N,cgiter=cgiter,accel="v_cycle",numlvls=numlvls)
        print("PCG with V cycle Gauss-Seidel error (w/o exact solve): " + str(la.norm(u_v3_pcg_v - u)/la.norm(u)))

        #u_v3_pcg_agg = pcg(A, x, f, N, accel="agg", numlvls=numlvls)
        #print("PCG with V cycle Gauss-Seidel error: " + str(la.norm(u_v3_pcg_v - u)/la.norm(u)))

    else:
        Npts = [i for i in range(32, 64, 4)]

        err_mg = [] # 2-level multigrid with Gauss-Seidel
        err_vc = [] # V-cycle with Gauss-Seidel
        err_pcg_mg = [] # PCG 2-level multigrid with Gauss-Seidel
        err_pcg_vc = [] # PCG V-cycle with Gauss-Seidel

        for N in Npts:
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


            u_mg,_,_ = multigrid_v3(A,x,f,1,2**2,N,1)
            u_vc = v_cycle(A,x,f,1,2**2,numlvls,1,exact_solve=True)
            u_pcg_mg = pcg(A,x,f,N,cgiter=cgiter,accel="mg",numlvls=numlvls)
            u_pcg_vc = pcg(A,x,f,N,cgiter=cgiter,accel="vc",numlvls=numlvls)

            err_mg.append(la.norm(u_mg -u)/la.norm(u))
            err_vc.append(la.norm(u_vc -u)/la.norm(u))
            err_pcg_mg.append(la.norm(u_pcg_mg -u)/la.norm(u))
            err_pcg_vc.append(la.norm(u_pcg_vc -u)/la.norm(u))

        # plot the errors
        plt.figure()
        plt.semilogy(Npts,err_mg,label = '2-level MG with GS', color = 'b', linestyle = '-')
        plt.semilogy(Npts,err_vc,label = 'V-cycle with GS', color = 'g', linestyle = '-')
        plt.semilogy(Npts,err_pcg_mg,label = 'PCG 2-level MG with GS', color = 'r', linestyle = '-')
        plt.semilogy(Npts,err_pcg_vc,label = 'PCG V-cycle with GS', color = 'y', linestyle = '-')
        plt.title('Error vs Size')
        plt.ylabel('Error (w/ exact solve)')
        plt.xlabel('Size')
        plt.legend()
        plt.show()

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
