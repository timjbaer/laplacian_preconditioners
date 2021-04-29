#ifndef __LOW_HOP_EMULATOR_H__
#define __LOW_HOP_EMULATOR_H__

#include "subemulator.h"

#define NUM_LEVELS 1
#define PENALTY    1.0
#define K          10

class LowHopEmulator {
  public:
    Subemulator * subems[NUM_LEVELS];
    Matrix<REAL> * G;
    int b;
    int mode;

    LowHopEmulator(Matrix<REAL> * A, int b, int mode=0);
    ~LowHopEmulator();

    Matrix<REAL> * sssp(int vertex);
    Matrix<REAL> * apsp();

  private:
    void hierarchy(Matrix<REAL> * A, int b);
    void collapse(Matrix<REAL> * A);
    Matrix<REAL> * sssp_hierarchy(int vertex);
    Matrix<REAL> * sssp_collapse(int vertex);
};

#endif
