import networkx as nx
import random as rand
import numpy as np
import scipy.sparse as sp

def random_graph(n,p=0.5,seed=0):
    directed = False
    G = nx.fast_gnp_random_graph(n,p,seed,directed)
    for (u,v) in G.edges():
        G.edges[u,v]['weight'] = rand.uniform(0, 1000)
    A = nx.to_scipy_sparse_matrix(G,format='csr')
    return A

def rmat_graph(n):
    A = sp.random(n,n,density=0.50,format='csr') * 100
    A = A @ A.T / 2
    A.setdiag(0)

    return A

def kronecker_graph(order):
    K = sp.csr_matrix((3,3))
    K[0,0] = 1
    K[0,1] = 1
    K[1,1] = 1
    K[1,2] = 1
    K[2,2] = 1

    K = K @ K.T / 2

    A = K
    for i in range(2, order + 1):
        A = sp.kron(A, K, format='csr')
    A.setdiag(0)

    return A

#   Unweighted graph
#   0 -- 1
#   | \  |
#   |  \ |
#   2 -- 3
#
#  Adjacency matrix
#  [[0, 1, 1, 1]
#   [1, 0, 0, 1]
#   [1, 0, 0, 1]
#   [1, 1, 1, 0]]
#
#  Shortest distance matrix
#  [[0, 1, 1, 1]
#   [1, 0, 2, 1]
#   [1, 2, 0, 1]
#   [1, 1, 1, 0]]
#
def sample_graph():
    A = sp.csr_matrix([[0, 1, 1, 1],
                       [1, 0, 0, 1],
                       [1, 0, 0, 1],
                       [1, 1, 1, 0]])

    return A
