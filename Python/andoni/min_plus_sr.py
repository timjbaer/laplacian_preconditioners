import numpy as np
import scipy.sparse as sp

def matpow2(A):
    """ Compute matrix square on (min, plus) semiring.

    Parameters
    ----------
    A: scripy.sparse (csr)
        Input matrix A
    Returns
    ----------
    B: scipy.sparse (csr)
        Output matrix B = AA
    """

    n = A.shape[0]

    B = A.copy()
    B_next = sp.csr_matrix((n,n))

    # B = B @ B.T # <- min-plus(B,B.T)
    # parallel for {
    for i in range(n):
        # parallel for {
        for j in range(n):
            if i == j:
                continue
            # parallel for {
            for k in range(n):
                if B[i,k] + B[k,j] == 0: # addid
                    continue
                elif B_next[i,j] == 0: # addid
                    B_next[i,j] = B[i,k] + B[k,j]
                else:
                    B_next[i,j] = min(B[i,k] + B[k,j], B_next[i,j])
            # }
        # }
    # }

    return B_next
