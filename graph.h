#ifndef __GRAPH_H__
#define __GRAPH_H__

#include <ctf.hpp>
#include <float.h>
#include <math.h>

using namespace CTF;
#define SEED 23
typedef int wht;
#define MAX_WHT (INT_MAX/2)

static Semiring<wht> MAX_TIMES_SR(0,
    [](wht a, wht b) {
      //return std::max(a, b);
      return a + b;
    },
    MPI_MAX,
    1,
    [](wht a, wht b) {
      return a * b;
    });

class Int64Pair {
  public:
    int64_t i1;
    int64_t i2;

    Int64Pair(int64_t i1, int64_t i2);

    Int64Pair swap();
};

void mat_set(Matrix<int>* matrix, Int64Pair index, int value = 1);

class Graph {
  public:
    int numVertices;
    vector<Int64Pair>* edges;

    Graph();

    Matrix<int>* adjacencyMatrix(World* world, bool sparse = false);
};

Matrix<int>* generate_kronecker(World* w, int order);

#endif
