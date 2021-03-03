#ifndef __GRAPH_IO_H__
#define __GRAPH_IO_H__

#include <ctf.hpp>
#include <float.h>
#include <math.h>

using namespace CTF;
#define SEED 23
typedef float REAL;
#define MAX_REAL (INT_MAX/2)

uint64_t read_graph_mpiio(int myid, int ntask, const char *fpath, uint64_t **edge, char ***led);

uint64_t read_graph(int myid, int ntask, const char *fpath, uint64_t **edge);

uint64_t read_metis(int myid, int ntask, const char *fpath, std::vector<std::pair<uint64_t, uint64_t> > &edges, int64_t * n, bool * e_weights, std::vector<REAL> &eweights);

#endif
