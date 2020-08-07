import numpy as np
import numpy.linalg as la
import scipy.sparse as sp
import scipy.linalg as sla
import scipy.sparse.linalg as spla

def separate_matrix(A):
    """Returns splitting of A=diag(d)-L-U via [L,d,U]"""
    assert(A.shape[0]==A.shape[1])
    return [-np.tril(A,k=-1), np.diag(A), -np.triu(A,k=1)]

"""
Relaxations
"""
def jacobi(A,u,f,w,numiter=100):
    """ runs @numiter iterations of Jacobi's smoothing
    
    Parameters
    ----------
    A,u,f : np.ndarray(s)
        set of arrays for the system Au=f
        vector in system Au=f
    w: float
        weight/dampening
    numiter : int 
        number of iterations
    
    Return
    ----------
    [v,err]: np.ndarray, float
        Jacobi iteration for v ~ A^-1 f and array of errors every iteration
    """
    [L,d,U] = separate_matrix(A)
    err = np.append(la.norm(-u,ord=np.inf), np.zeros(maxiter))

    # iteration matrix
    Dinv = np.diag(1/d)
    R_J = np.dot(Dinv, L+U)
    R = (1-w)*np.eye(A.shape[0]) + w*R_J; Df = w*np.dot(Dinv,f)

    for i in range(numiter):
        u = np.dot(R,u) + Df
        err[i+1] = la.norm(-u,ord=np.inf)
    return u,err

# Jacobi Smoother
def wtd_jacobi(A,x,b,w,numiter):
    """ another implemenation of Jacobi smoother
    
    Parameters
    ----------
    A,x,b : np.ndarray(s)
        set of arrays for the system Ax=b
    w: float
        weight/dampening
    numiter : int 
        number of iterations
    
    Return
    ----------
    [v,err]: np.ndarray, float
        Jacobi iteration for v ~ A^-1 f and array of errors every iteration
    """
    D = A.diagonal()
    R = A.copy()
    R.setdiag(0)
    u = x.copy()
    for _ in range(numiter):
        u = w * (D**(-1) * (b-R@u)) + (1-w) * u
    return u

def _gauss_seidel(A,u,f,maxiter=100):
    """ 

    Parameters
    ----------
    A,u,f : np.ndarray(s)
        vectors for the system system Au=f
    maxiter : int
        maximum number of iterations

    Return
    ----------
    v : np.ndarray
        Jacobi iteration for v ~ A^-1 f
    """
    [L,d,U] = separate_matrix(A)
    err = np.append(la.norm(-u,ord=np.inf), np.zeros(maxiter))

    # iteration matrix
    DmU = np.diag(d)-U
    for i in range(maxiter):
        u = la.solve(DmU, np.dot(L,u) + f)
        err[i+1] = la.norm(u,ord=np.inf)
    return u,err

# Gauss-Seidel Smoother
def gauss_seidel(A,x,b,w,numiter):
    """ Another implemenation of gauss-seidel

    Parameters
    ----------
    A,u,f : np.ndarray(s)
        vectors for the system system Au=f
    maxiter : int
        maximum number of iterations
    w : float
        dampening factor

    Return
    ----------
    v : np.ndarray
        Jacobi iteration for v ~ A^-1 f
    """
    L = sp.tril(A,0).tocsr()
    U = sp.triu(A,1).tocsr()
    u = x.copy()
    for _ in range(numiter):
        u = spla.spsolve(L,b-U@u)
    return u
