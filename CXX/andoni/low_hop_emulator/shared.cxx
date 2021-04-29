#include "shared.h"

int64_t are_matrices_different(Matrix<REAL> * A, Matrix<REAL> * B)
{
  Scalar<int64_t> s;
  s[""] += Function<REAL,REAL,int64_t>([](REAL a, REAL b){ return (int64_t) fabs(a - b) >= EPSILON; })((*A)["ij"],(*B)["ij"]);
  return s.get_val();
}
