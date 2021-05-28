#ifndef __LOW_HOP_EMULATOR_H__
#define __LOW_HOP_EMULATOR_H__

#include "subemulator.h"

#define NUM_LEVELS 4
#define PENALTY    1.0
#define K          10

class LowHopEmulator {
  public:
    int n; // number of vertices
    int b0; // initial ball size
    int mode; // hierarchial (mode=1) or collapsed (mode=0) low hop emulator
    World * w; // CTF world
    Subemulator * subems[NUM_LEVELS+1]; // subemulators, including input graph
    Matrix<REAL> * G; // collapsed low hop emulator (mode=0)

    LowHopEmulator(Matrix<REAL> * A, int b0_, int mode_=0);
    ~LowHopEmulator();

    Matrix<REAL> * sssp(int vertex);
    Matrix<REAL> * apsp();

  private:
    void hierarchy(Matrix<REAL> * A);
    void collapse(Matrix<REAL> * A);
    Matrix<REAL> * sssp_hierarchy(int vertex);
    Matrix<REAL> * sssp_collapse(int vertex);
};

#endif
