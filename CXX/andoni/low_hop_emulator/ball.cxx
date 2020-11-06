#include "ball.h"

Matrix<REAL> * ball(Matrix<REAL> * A, int64_t b) { // A should be on (min, +) semiring
  Matrix<REAL> * B = new Matrix<REAL>(*A);
  for (int i = 0; i < log2(B->nrow); ++i) {
    (*B)["ij"] += (*B)["ik"] * (*B)["kj"];
  }
  return B;
}
