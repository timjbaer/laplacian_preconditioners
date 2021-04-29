#ifndef __TEST_H__
#define __TEST_H__

#include "../shared.h"
#include "../../../graph.h"

Matrix<REAL> * correct_dist(Matrix<REAL> * A, int b, char * name="input");

char* getCmdOption(char ** begin,
                   char ** end,
                   const   std::string & option);

Matrix<REAL> * get_graph(int const in_num, char** input_str, World & w);

void perturb(Matrix<REAL> * A);

#endif
