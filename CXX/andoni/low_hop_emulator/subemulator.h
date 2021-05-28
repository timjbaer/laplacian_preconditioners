#ifndef __SUBEMULATOR_H__
#define __SUBEMULATOR_H__

#include "ball.h"

#define SAMPLE_PROB 0.50

class Subemulator {
  public:
    int n; // number of vertices
    int b; // ball size
    World * w; // CTF world
    Matrix<REAL> * H; // subemulator
    Vector<bpair> * q; // leaders
    Matrix<REAL> * B; // ball of subemulator

    Subemulator(Matrix<REAL> * A, Matrix<REAL> * B_A, int b_);
    Subemulator(Subemulator * A, int b_); // reuse ball information from previous subemulator
    Subemulator(Matrix<REAL> * H_, Vector<bpair> * q_, int b_);

    ~Subemulator();

    Vector<int> * samples(Matrix<REAL> * B_A);

    void connects(Matrix<REAL> * A, Matrix<REAL> * B_A, Vector<int> * S);
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
