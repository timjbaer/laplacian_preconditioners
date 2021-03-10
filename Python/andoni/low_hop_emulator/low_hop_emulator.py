"""
Idea:         Given a graph, compute a sparser (w.r.t. vertices) graph that approximates distances well.

Why:          Main kernel necessary for efficient parallel Bourgain's embedding.

Dependencies: Ball.

Resources:    Caleb/Tim's slides and Andoni et. al section 3 Low Hop Emulator.
"""

import sys
sys.path.append("..")
import argparse

import numpy as np
import scipy.sparse as sp
from graph import random_graph, rmat_graph, kronecker_graph, sample_graph
from ball import ball
import min_plus_sr

def _subemulator(A, b):
    n = A.shape[0]

    ## construct vertices ##
    S = np.array([0 for i in range(n)])
    # parallel for {
    for i in range(n):
        rand = np.random.random()
        if rand < min(50.0 * np.log(n) / b, 0.5):
            S[i] = 1
    # }

    ball_ = ball(A,b) # ball is not fully symmetric
    # if __debug__: print("ball->\n{}".format(ball_.todense()))

    covered = set() # see slide 9
    # parallel for {
    for i in range(n):
        if S[i] == 1:
            _, col = ball_.getrow(i).nonzero()
            # parallel for {
            for j in col:
                covered.add(j)
            # }
    # }

    # parallel for {
    for i in range(n):
        if S[i] == 0 and i not in covered:
            S[i] = 1
    # }
    # if __debug__: print("S ->\n{}".format(S))

    q = np.array([0 for i in range(n)])
    # parallel for {
    for i in range(n):
        _, col = ball_.getrow(i).nonzero()
        # parallel for {
        for j in col:
            if ball_[i,j] == 0: # addid
                continue
            elif ball_[i,q[i]] == 0: # addid
                q[i] = j
            else:
                if ball_[i,j] < ball_[i,q[i]]:
                    q[i] = j
        # }
    # }
    # if __debug__: print("q->\n{}".format(q))

    ## construct edges ##
    H = sp.csr_matrix((n,n))

    # parallel for {
    for i in range(n):
        _, col = A.getrow(i).nonzero()
        # parallel for {
        for j in col:
            assert(A[i,j] > 0 and ball_[j,q[j]] > 0)
            if ball_[q[i],i] == 0 or q[i] == q[j]:
                continue
            if H[q[i],q[j]] == 0:
                H[q[i],q[j]] = ball_[q[i],i] + A[i,j] + ball_[j,q[j]]
                H[q[j],q[i]] = ball_[q[i],i] + A[i,j] + ball_[j,q[j]]
            else:
                H[q[i],q[j]] = min(ball_[q[i],i] + A[i,j] + ball_[j,q[j]], H[q[i],q[j]])
                H[q[j],q[i]] = min(ball_[q[i],i] + A[i,j] + ball_[j,q[j]], H[q[i],q[j]])
        # }
    # }

    # parallel for {
    for i in range(n):
        _, col = ball_.getrow(i).nonzero()
        # parallel for {
        for j in col:
            if i == j:
                continue
            assert(ball_[j,q[j]] > 0)
            if ball_[q[i],i] == 0 and A[q[i], q[j]] == 0 or q[i] == q[j]:
                continue
            elif H[q[i], q[j]] == 0:
                H[q[i], q[j]] = ball_[q[i], i] + A[i,j] + ball_[j, q[j]]
                H[q[j], q[i]] = ball_[q[i], i] + A[i,j] + ball_[j, q[j]]
            else:
                H[q[i], q[j]] = min(ball_[q[i], i] + A[i,j] + ball_[j, q[j]], H[q[i], q[j]])
                H[q[j], q[i]] = min(ball_[q[i], i] + A[i,j] + ball_[j, q[j]], H[q[i], q[j]])
        # }
    # }

    # if __debug__: print("H->\n{}".format(H.todense()))

    ## properties ##
    logn = int( np.ceil(np.log(n)/np.log(2)) )

    D_H = H.copy()
    for itr in range(logn):
        D_H = min_plus_sr.matpow2(D_H)
    # if __debug__: print("D_H->\n{}".format(D_H.todense()))

    D = A.copy()
    for itr in range(logn):
        D = min_plus_sr.matpow2(D)
    # if __debug__: print("D->\n{}".format(D.todense()))

    """
    if __debug__:
        for i in S:
            for j in S:
                if i == j:
                    continue
                assert(D[i,j] <= D_H[i,j] <= 8 * D[i,j])

        for i in range(n):
            for j in range(n):
                assert(D_H[q[i],q[j]] <= D[i,q[i]] + 22 * D[i,j] + D[j,q[j]])

        print("preserves distances well!")

        print("fraction of number of edges compared to original graph: {}".format(H.count_nonzero() / A.count_nonzero()))
    """

    return [H, q]

class dist_oracle:
    def __init__(self,A,k): # preprocess
        self.t = 0
        self.D_H = []
        self.q_H = []
        self.B_H = []

        n = A.shape[0]

        H = A.copy()
        # b = max(np.ceil((75*np.log2(n))**2), n**(1/(2*k)))
        b = 3

        while H.shape[0] >= b:
            H,q = _subemulator(H,b)
            B = ball(H,b)
            self.B_H.append(B)
            for i in range(H.shape[0]): # include leader
                B[i,q[i]] = H[i,q[i]]
            self.D_H.append(B)
            self.q_H.append(q)

            b = int(np.ceil((b ** 1.25)))
            self.t += 1

        D = H
        for i in range(int(np.ceil(np.log2(H.shape[0])))):
            D = min_plus_sr.matpow2(D)
        self.D_H.append(D)
        self.q_H.append([H.indices[0] for i in range(H.shape[0])])

    def _query(self,i,j):
        l = 0

        d = 0
        while i in [k for k in range(10)]:
        # while i not in self.B_H[l][j] and j not in self.B_H[l][i]:
            q_i = self.q_H[l][i]
            q_j = self.q_H[l][j]
            d += self.D_H[l][i,q_i] + self.D_H[l][j,q_j]
            i = q_i
            j = q_j
            l += 1
        d += 0 # TODO: fix
        return d

def low_hop_emulator(A,b):
    t = _pre_proc(A, b)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-ner", help="n for Erdos-Renyi graph")
    parser.add_argument("-nrmat", help="n for RMAT graph")
    parser.add_argument("-nk", help="n for Kronecker graph")
    parser.add_argument("-b", help="ball size")

    args = parser.parse_args()

    if args.ner:
        A = random_graph(int(args.ner))
    elif args.nrmat:
        A = rmat_graph(int(args.nrmat))
    elif args.nk:
        A = kronecker_graph(int(int(args.nk) ** (1/3)))
    else:
        A = sample_graph()
    # if __debug__: print("A->\n{}".format(A.todense()))

    if args.b:
        b = int(args.b)
    else:
        b = int(np.log2(A.shape[0]))
    # if __debug__: print("b->\n{}".format(b))

    # _subemulator(A,b)
    k = np.random.uniform(low=0.5, high=0.5 * np.log2(A.shape[0]))
    dist_oracle_ = dist_oracle(A, k)
    print(dist_oracle_._query(1,2))

if __name__ == '__main__':
    main()
