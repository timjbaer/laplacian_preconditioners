#ifndef __SUBEMULATOR_H__
#define __SUBEMULATOR_H__

#include "ball.h"

class Subemulator {
  public:
    Matrix<REAL> * H; // subemulator
    Vector<bpair> * q; // leaders
    int b; // ball size
    Matrix<REAL> * B; // ball

    Subemulator(Matrix<REAL> * A, int b);

    ~Subemulator();

    Vector<int> * samples();

    void connects(Matrix<REAL> * A, Vector<int> * S);
};

class DistOracle {

};

class LowHopEm {

};

static Semiring<int> MAX_TIMES_SR(0,
    [](int a, int b) {
      return std::max(a, b);
    },
    MPI_MAX,
    1,
    [](int a, int b) {
      return a * b;
    });

static Monoid<REAL> MAX_MONOID(0,
    [](REAL a, REAL b) {
      return std::max(a,b);
    },
    MPI_MAX
    );

#endif
