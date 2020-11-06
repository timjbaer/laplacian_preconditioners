# Usage
1. in `../../generator` directory, run `make -f Makefile.BFS.mpi` to compile `libgraph_generator_mpi.a`
2. `make`
3. `mpirun -np [NUM_THREADS] ./test [OPTS]`

| Graph     | OPTS  |
| --------- | ----- |
| Kronecker | -k    |
| RMAT      | -E -S |
| METIS     | -f -n |
