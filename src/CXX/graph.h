#ifndef __GRAPH_H__
#define __GRAPH_H__

#include "graph_io.h"

#include <ctf.hpp>
#include <float.h>
#include <math.h>

using namespace CTF;
#define SEED 23
typedef float REAL;
#define MAX_REAL (INT_MAX/2)

static Semiring<REAL> MAX_TIMES_SR(0,
    [](REAL a, REAL b) {
      //return std::max(a, b);
      return a + b;
    },
    MPI_SUM,
    1,
    [](REAL a, REAL b) {
      return a * b;
    });

class Int64Pair {
  public:
    int64_t i1;
    int64_t i2;

    Int64Pair(int64_t i1, int64_t i2);

    Int64Pair swap();
};

class Graph {
  public:
    int numVertices;
    vector<Int64Pair>* edges;

    Graph();

    Matrix<REAL>* adjacencyMatrix(World* world, bool sparse = false);
};

Matrix<REAL>* generate_kronecker(World* w, int order);

Matrix <REAL> * read_matrix(World  &     dw,
                         int          n,
                         const char * fpath,
                         bool         remove_singlets,
                         int *        n_nnz,
                         int64_t      max_ewht = MAX_REAL);

Matrix <REAL> gen_uniform_matrix(World & dw,
                                int64_t n,
                                double  sp,
                                int64_t  max_ewht = MAX_REAL);

#endif
