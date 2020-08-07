import numpy as np
import scipy.sparse as sp

def poisson_2d(n):
    """ Given a size n, return a sparse matrix for 2D Poisson problem """
    two_d_poisson_stencil = np.asarray([
        [0, 1, 0],
        [1,-4, 1],
        [0, 1, 0]
    ])
    h = 1/(n-1)
    A = (1/h**2) * twod_model_with_stencil(
        two_d_poisson_stencil, n)
    return A

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


# model problem
def oned_model_with_stencil(stencil,n):
    """ Given a 1D stencil, constructs model problem via banded, diagonal amtrix

    As an example, consider the 1D stencil [-1, 2, -1], meaning we want the set
    of equations (w/ some boundary conditions, e.g. u_0=u_n=0.) to satisfy:

        -u_{i-1} + 2u_i - u_{i+1} = f_i.

    The system of equations can be written via

        (1/(n-1)) * Tu=f,

    where tridiag(-1,2,-1) and n is the number of unknowns.

    Parameters
    ----------
    stencil : numpy.ndarray
        1D stencil. Must be of len = 3 (TODO: Should we allow len > 3?)
    n : int
        size of model problem

    Returns
    ----------
    scipy.sparse.csr.csr_matrix
        Matrix for model problem
    """
    assert(len(stencil)==3)
    return sp.diags(
        diagonals=stencil, # diagonal elements
        offsets=[-1,0,1], # location of diags
        shape=(n,n),
        format="csr"
    )

# model problem
def twod_model_with_stencil(stencil,n):
    """ Given 2D stencil, constructs model problem via banded, diagonal matrix

    As an example, consider 9-point uniform 2D stencil 

        [-1, -1, -1]
        [-1,  8, -1]
        [-1, -1, -1]

    which like the 1D stencil, but with nine terms due to the 2D nature.  We
    can think about each row as a different value along the y dimension and
    each columnas a different value along the x dimension. Such a system has an
    analogous interpretation for a block-matrix: here, each block corresponds
    to a different y-coordinate and the coordinates within each block for the
    x-coordinate. 

    Parameters
    ----------
    stencil : numpy.ndarray
        2D stencil. Must be of dimension 3
    n : int
        size of model problem

    Returns
    ----------
    scipy.sparse.csr.csr_matrix
        Matrix for model problem
    """
    assert( (3,3) == stencil.shape )
    # which block diagonal
    I_sup,I_main,I_sub = [
        sp.diags(diagonals=[1], offsets=[-1+i], shape=(n,n), format="csr") for i in range(3)
    ]
    # matrix to put on each block diagonal
    B_sup,B_main,B_sub = [
        oned_model_with_stencil(stencil[i,:],n) for i in range(3)
    ]
    
    return sp.kron(I_sup,B_sup) + sp.kron(I_main,B_main) + sp.kron(I_sub,B_sub)
