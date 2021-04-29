#ifndef __SHARED_H__
#define __SHARED_H__

#include <ctf.hpp>
#include <float.h>
#include <math.h>
#include <ctime>

#ifdef CRITTER
#include "critter.h"
#else
#define TAU_FSTART(ARG)
#define TAU_FSTOP(ARG)
#endif

using namespace CTF;
typedef float REAL;
#define SEED 23
#define MAX_REAL  (INT_MAX/2)
#define EPSILON   0.01

int64_t are_matrices_different(Matrix<REAL> * A, Matrix<REAL> * B);

#endif
