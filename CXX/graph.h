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

static Semiring<REAL> PLUS_TIMES_SR(0,
    [](REAL a, REAL b) {
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
    ~Graph();

    Matrix<REAL>* adjacencyMatrix(World* world, bool sparse = true);
};

uint64_t gen_graph(int scale, int edgef, uint64_t seed, uint64_t **edges);

Matrix<REAL> * preprocess_graph(int64_t            n,                                         
                               World &        dw,                                       
                               Matrix<REAL> * A_pre,                                    
                               bool           remove_singlets,                          
                               int64_t *          n_nnz,                                    
                               int64_t        max_ewht=1);                              
                                                                                      
Matrix<REAL> * read_matrix(World  &     dw,                                              
                           int64_t          n,                                              
                           const char * fpath,                                          
                           bool         remove_singlets,                                
                           int64_t *        n_nnz,                                          
                           int64_t      max_ewht=1);                                    
                                                                                      
Matrix<REAL> * gen_rmat_matrix(World  & dw,                                              
                              int      scale,                                          
                              int      ef,                                             
                              uint64_t gseed,                                          
                              bool     remove_singlets,                                
                              int64_t *    n_nnz,                                          
                              int64_t  max_ewht=1);                                    
                                                                                      
Matrix <REAL> * gen_uniform_matrix(World & dw,
                                int64_t n,
                                bool remove_singlets,
                                int64_t * n_nnz,
                                double  sp=0.01,
                                int64_t  max_ewht=1);
                                                                                      
Matrix<REAL>* generate_kronecker(World* w, int order);    

#endif
