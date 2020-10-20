import numpy as np
import scipy.sparse as sp
import graph
from low_hop_emulator.ball import ball

def main():
    n = 16
    b = 7
    A = graph.random_graph(n,0.8)
    ball(A,b)

if __name__ == '__main__':
    main()
