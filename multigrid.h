#ifndef __MULTIGRID_H__
#define __MULTIGRID_H__

#include "graph.h"

typedef float REAL;

void smooth_jacobi(Matrix<REAL> & A, Vector<REAL> & x, Vector <REAL> & b, int nsm);
void vcycle(Matrix<REAL> & A, Vector<REAL> & x, Vector<REAL> & b, Matrix<REAL> * P, Matrix<REAL> * PTAP, int64_t N, int nlevel, int * nsm);

void setup(Matrix<REAL> & A, Matrix<REAL> * T, int N, int nlevel, Matrix<REAL> * P, Matrix<REAL> * PTAP);
void setup_laplacian(int64_t         n,
                     int             nlvl,
                     REAL            sp_frac,
                     int             ndiv,
                     int             decay_exp,
                     Matrix<REAL>  & A,
                     Matrix<REAL> *& P,
                     Matrix<REAL> *& PTAP,
                     World &         dw);

#endif
