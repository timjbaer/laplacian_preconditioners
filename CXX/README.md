# Laplacian Preconditioning via AMG

## Usage
Currently supports METIS graph files and Kronecker graphs (built-in).

`mpirun -np 4 ./test -k 2`
`mpirun -np 4 ./test -f data/weighted_graph.gr -n 767`
