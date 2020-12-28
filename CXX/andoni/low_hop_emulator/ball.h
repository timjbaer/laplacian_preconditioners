#ifndef __BALL_H__
#define __BALL_H__

#include <ctf.hpp>
#include <float.h>
#include <math.h>

using namespace CTF;
#define SEED 23
typedef float REAL;
#define MAX_REAL (INT_MAX/2)

static Semiring<REAL> MIN_PLUS_SR(MAX_REAL,
    [](REAL a, REAL b) {
      return std::min(a, b);
    },
    MPI_MIN,
    0,
    [](REAL a, REAL b) {
      return a + b;
    });

struct bpair {
  int col;
  int dist;
};

template<int b>
class bvector {
  public: 
    bpair closest_neighbors[b];

    bvector() { 
      for (int i = 0; i < b; ++i) {
        closest_neighbors[i].col = -1;
        closest_neighbors[i].dist = MAX_REAL;
      }
    }
};

template<int b>
void bvector_red(bvector<b> const * x,
                 bvector<b> * y,
                 int n){
  std::set<int> cols;
  int i,j,k;
  i = j = k = 0;
  while (k < b) { // TODO: write as parallelizable for loop
    if (x[i].dist < y[j].dist) {
      {
        if (cols.find(x[i].col) != cols.end()) { cols.insert(x[i].col); }
        else { ++i; continue; }
      }
      y->col[k] = x[i]->col;
      y->dist[k] = x[i]->col;
      ++i;
      ++k;
    } else {
      {
        if (cols.find(y[i].col) != cols.end()) { cols.insert(y[i].col); }
        else { ++j; continue; }
      }
      ++j;
      ++k;
    }
  }
  std::sort(y, y + b,
            [](bvector<b> const & first, bvector<b> const & second) -> bool
              { return first.dist < second.dist; }
            );  // TODO: sort on insert?
}

template<int b>
Monoid< bvector<b> > get_bvector_monoid() {
  MPI_Op obvector;

  MPI_Op_create(
      [](void * x, void * y, int * n, MPI_Datatype*){
        bvector_red<b>((bvector<b>*) x, (bvector<b>*) y, *n);
      },
      1,
      &obvector);

  Monoid< bvector<b> > m(bvector<b>(),
      [](bvector<b> x, bvector<b> y) {
          bvector<b> z;
          std::set<int> cols;
          int i,j,k;
          i = j = k = 0;
          while (k < b) { // TODO: write as parallelizable for loop
            if (x[i].dist < y[j].dist) {
              {
                if (cols.find(x[i].col) != cols.end()) { cols.insert(x[i].col); }
                else { ++i; continue; }
              }
              z->col[k] = x[i]->col;
              z->dist[k] = x[i]->col;
              ++i;
              ++k;
            } else {
              {
                if (cols.find(y[i].col) != cols.end()) { cols.insert(y[i].col); }
                else { ++j; continue; }
              }
              z->col[k] = y[i]->col;
              z->dist[k] = y[i]->col;
              ++j;
              ++k;
            }
          }
          std::sort(z, z + b,
                    [](bvector<b> const & first, bvector<b> const & second) -> bool
                      { return first.dist < second.dist; }
                    ); // TODO: sort on insert?
          return z;
      },
      obvector
      );

  return m;
}

Matrix<REAL> * ball(Matrix<REAL> * A, int64_t b);

template<int b>
Vector<bvector<b>> * ball_bvector(Matrix<REAL> * A) { // A should be on B_VECTOR_MONOID
  Vector<bvector<b>> * B = new Vector<bvector<b>>(*A);
  for (int i = 0; i < b; ++i) {
    (*B)["i"] += (*A)["ij"] + (*B)["j"];
  }
  return B;
}

#endif
