#ifndef __TEST_H__
#define __TEST_H__

#include "../shared.h"
#include "../../../graph.h"

// ==================================================================
// Shared Functions for Test Cases
// ==================================================================

// compute true shortest paths distances between all nodes in A
// using Bellman Ford
Matrix<REAL> * correct_dist(Matrix<REAL> * A, int b, char * name="input");

// used for flags parser
char* getCmdOption(char ** begin,
                   char ** end,
                   const   std::string & option);

// input graph constructor
Matrix<REAL> * get_graph(int const in_num, char** input_str, World & w);

// adds random values between [0,1] to all values in A
void perturb(Matrix<REAL> * A);

#endif
