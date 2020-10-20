"""
Idea:         Given a vertex $v$, compute $b$ closest vertices to $v$.

Why:          Main kernel necessary to construct low hop emulators.

Dependencies: None.

Resources:    Mina's ball computation notes, Caleb/Tim's slides, and Andoni et. al appendix D.
"""

import numpy as np
import scipy.sparse as sp

def _closest_b(A,b):
    """ Given matrix where ith row contains neighbors, calculate the closest
        b of them and save into the ith row of another matrix
    """
    n = A.shape[0]
    data = np.array([])
    indices = np.array([])
    indptr = np.array([0])
    # parallel for {
    for i in range(n):
        # want to only pull nonzeros (at most b of them except for first round)
        neighbors = A.getrow(i)
        neighbors = neighbors.todense()
        neighbors = np.asarray(neighbors).reshape(-1)
        v_idx = np.argsort(neighbors,kind='mergesort') 
        v = neighbors[v_idx]
        first_edge_idx = np.argmax(v > 0.0)
        closest_b_neighbors_idx = v_idx[first_edge_idx : min(first_edge_idx + b, n) ]

        # can do balanced tree to compile into single list
        data = np.append(data, neighbors[closest_b_neighbors_idx])
        indices = np.append(indices, closest_b_neighbors_idx)
        indptr = np.append(indptr, len(data))
    # } 
    L = sp.csr_matrix((data,indices,indptr),shape=(n,n))
    return L

def ball(A,b):
    """ Computes the balls for each vertex. Assuming parallel matrix operations,
        runs in log(n) depth and O(n b^2 log(n)) work.

    Parameters
    ----------
    A : scipy.sparse (csr)
        Weighted adjacency matrix for an undirected graph G
    b : int
        Number of closest neighbors
    
    Returns
    ----------
    N : np.ndarray
        Matrix where the ith row contains ith vertex's closest <=b vertices
    """
    print("A ->\n{}".format(A.todense()))
    n = A.shape[0]

    # Compute b closest neighbors
    L = _closest_b(A,b)

    # Run truncated path doubling
    logn = int( np.ceil(np.log(n)/np.log(2)) )
    print("L ->\n{}".format(L.todense()))
    return L

    for i in range(logn):
        L_next = L@L.T # <- min-plus(L,L.T)

        L = _closest_b(L_next,b)
        print("L ->\n{}".format(L.todense()))
