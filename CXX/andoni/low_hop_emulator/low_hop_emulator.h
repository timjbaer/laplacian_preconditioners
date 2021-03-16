#ifndef __LOW_HOP_EMULATOR_H__
#define __LOW_HOP_EMULATOR_H__

#include "ball.h"

class Subemulator {
  Matrix<REAL> * H;
  Vector<int>  * q;

  Subemulator(Matrix<REAL> * A, int b);

  ~Subemulator();

  Vector<int> * samples(Matrix<REAL> * A, int b);

  void connects(Matrix<REAL> * A, Vector<int> * S, int b);
};

class DistOracle {

};

class LowHopEm {

};

#endif
