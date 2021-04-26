#ifndef __SUBEMULATOR_H__
#define __SUBEMULATOR_H__

#include "ball.h"

#define SAMPLE_PROB 0.5

class Subemulator {
  public:
    Matrix<REAL> * H; // subemulator
    Vector<bpair> * q; // leaders
    Matrix<REAL> * B; // ball // TODO: delete, pass as parameter instead
    int b; // ball size

    Subemulator(Matrix<REAL> * A, int b_);

    Subemulator(Matrix<REAL> * H_, Vector<bpair> * q_, int b_);

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

static Semiring<REAL> MIN_TIMES_SR(MAX_REAL,
    [](REAL a, REAL b) {
      return std::min(a, b);
    },
    MPI_MIN,
    1,
    [](REAL a, REAL b) {
      return (fabs(a - MAX_REAL) >= EPSILON && fabs(b - MAX_REAL) >= EPSILON) ? a * b : MAX_REAL;
    });

#endif
