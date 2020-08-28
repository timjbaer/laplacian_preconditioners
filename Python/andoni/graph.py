import networkx as nx

def random_graph(n,p=0.5,seed=0):
    directed = False
    G = nx.fast_gnp_random_graph(n,p,seed,directed)
    A = nx.to_scipy_sparse_matrix(G,format='csr')
    return A
