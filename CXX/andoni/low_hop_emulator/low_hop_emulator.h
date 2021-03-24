#ifndef __LOW_HOP_EMULATOR_H__
#define __LOW_HOP_EMULATOR_H__

#include "subemulator.h"

#define NUM_LEVELS 16
#define PENALTY    1.0
#define K          0.5

class LowHopEmulator {
  public:
    Matrix<REAL> * G;
    int b;

    LowHopEmulator(Matrix<REAL> * A, int b, int mode=0);
};

#endif
