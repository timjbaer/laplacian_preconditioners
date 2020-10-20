"""
Idea:         Given a graph, compute a sparser (w.r.t. vertices) graph that approximates distances well.

Why:          Main kernel necessary for efficient parallel Bourgain's embedding.

Dependencies: Ball.

Resources:    Caleb/Tim's slides and Andoni et. al section 3 Low Hop Emulator.
"""

import numpy as np
import scipy.sparse as sp
import graph
import ball

INT_MAX = 10**10

def _pre_proc(A):
    pass

def _subemulator(A, b):
    n = A.shape[0]

    ## construct vertices ##
    S = np.array([INT_MAX for i in range(n)])
    # parallel for {
    for i in range(n):
        rand = np.random.random()
        #if rand < min(50.0 * np.log(n) / b, 0.5):
        if rand < 0.2:
            S[i] = 1
    # }

    # on (min, times), addid is \infty
    ball_ = ball.ball(A,b)
    ball_ = ball_.todense()
    for i in range(n):
        for j in range(n):
            if ball_[i,j] == 0:
                ball_[i,j] = INT_MAX
    ball_ = sp.csr_matrix(ball_)


    print("ball")
    print(ball_.todense())

    print("S")
    print(S)

    # S_i += not ball_{ij} * S_i
    for i in range(n):
        b_neighbors = ball_.getrow(i)
        b_neighbors = b_neighbors.todense()
        b_neighbors = np.asarray(b_neighbors).reshape(-1)

        if i == 1:
            print("itr")
            print(b_neighbors)
            print(S)
            print("out")
            print(np.multiply(b_neighbors,S))
            print()

        S[i] = min(S[i], int(not(min(np.multiply(b_neighbors,S)))))

    print("S")
    print(S)

def low_hop_emulator(A):
    t = _pre_proc(A)


def main():
    n = 10
    b = 2
    A = graph.random_graph(n,0.8)
    _subemulator(A, b)

if __name__ == '__main__':
    main()
