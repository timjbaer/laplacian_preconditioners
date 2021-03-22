#ifndef __TEST_H__
#define __TEST_H__

#include <ctime>
#include <ctf.hpp>
#include <float.h>
#include <math.h>

#ifdef CRITTER
#include "critter.h"
#else
#define TAU_FSTART(ARG)
#define TAU_FSTOP(ARG)
#endif

#include "../../../graph.h"

using namespace CTF;
typedef float REAL;
#define SEED 23
#define MAX_REAL  (INT_MAX/2)
#define EPSILON   0.01

char* getCmdOption(char ** begin,
                   char ** end,
                   const   std::string & option);

Matrix<REAL> * get_graph(int const in_num, char** input_str, World & w);

void perturb(Matrix<REAL> * A);

int64_t are_matrices_different(Matrix<REAL> * A, Matrix<REAL> * B);

#endif
