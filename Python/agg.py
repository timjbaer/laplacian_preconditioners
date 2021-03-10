import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import matplotlib.pyplot as plt

from relaxations import wtd_jacobi, gauss_seidel

def pairs(A, beta=0.20): # TODO: iterate over CSR layout
    """Pairwise aggregator
    @param A: SDD matrix
    @param beta: tolerance

    reference: An aggregation-base algebraic multigrid method (Notay)
    """
    G = []

    class Node:
        def __init__(self, i, S, m):
            self.i = i
            self.S = S
            self.m = m

    U = {} # node i -> {"S": S[i], "m": m[i]}

    n = A.shape[0]
    S = []
    for i in range(n):
        S_i = []
        max_aik = max([abs(aik) for aik in A[i]]) # max_{a_{ik} < 0} |a_{ik}|
        for j in range(n):
            if i != j and abs(A[i, j]) > beta * max_aik: # j \notin U\{i} : a_{ij} < -\beta * maxaik
                S_i.append(j)
        S.append(S_i)

    m = []
    for i in range(n):
        m.append(len(set([j for j in range(n) if i in S[j]]))) # m_i = |\{j : i \in S_j\}|

    for i in range(n):
        U[i] = Node(i, S[i], m[i])

    while len(U) != 0:
        min_i = list(U)[0]
        for i in U: # i \in u with minimal m_i 
            if U[i].m < U[min_i].m:
                min_i = i
        i = min_i

        min_j = 0
        for k in U: # j \in U such that a_{ij} = min_{k \in U} a_{ik}
            if A[i,k] < A[i, min_j]:
                min_j = k
        j = min_j

        G_nc = []
        del_j = False
        if j in U[i].S: # if j \in S_i: G_{nc} = {i, j}
            G_nc.append(i)
            G_nc.append(j)
            del_j = True
        else: # otherwise: G_{nc} = {i}
            G_nc.append(i)
        for k in G_nc:
            for l in U[k].S: # m_l = m_l - 1 for l in S_k
                if l in U:
                    U[l].m -= 1
                    if k in U[l].S:
                        U[l].S.remove(k)
        G.append(G_nc)

        del U[i]
        if del_j:
            del U[j]

    return G

def pairwise_agg(A): # TODO: iterate over CSR layout
    """Pairwise aggregation scheme based on strong connections
    @param A: SDD matrix
    """

    A = A.toarray()
    n = A.shape[1]
    G = pairs(A)
    nc = len(G)

    P = np.zeros((n, nc), dtype=float)
    for i in range(n):
        for j in range(nc):
            if i in G[j]:
                P[i, j] = 1
            else:
                P[i, j] = 0
    P = sp.csr_matrix(P)

    return P

# multigrid 3 level
def agg_multigrid(A,f,x,w,iter,N,ind): # TODO: refactor multigrid to accept a restriction operator P?
    # form multigrid operators for all levels
    P1_2D = pairwise_agg(A)
    A1 = P1_2D.T@  A @P1_2D
    nc1 = P1_2D.shape[1]

    P2_2D = pairwise_agg(A1)
    A2 = P2_2D.T@ A1 @P2_2D

    ## level 0 ##
    smooth = gauss_seidel if ind==1 else wtd_jacobi
    u = smooth(A,x,f,w,iter) # pre-smoothing
    r = f - A@u # form residual
    f1 = P1_2D.T@r # restrict

    ## Level 1 ##
    #e = np.zeros(nc1**2)
    e = np.zeros(P1_2D.shape[1])
    u1 = smooth(A1,e,f1,w,iter) # smoothing
    r1 = f1 - A1@u1 # form residual
    f2 = P2_2D.T@r1 # restrict

    ## Level 2 ##
    u2 = sla.spsolve(A2,f2) #solve system

    ## Level 1 ##
    u1 = u1 + P2_2D@u2 # interpolate and correct previous solution
    u1 = smooth(A1,u1,f1,w,iter) # smoothing

    ## Level 0 ##
    u = u + P1_2D@u1 # interpolate and correct previous solution

    u = smooth(A,u,f,w,iter) # post-smoothing

    return u, A1, A2
