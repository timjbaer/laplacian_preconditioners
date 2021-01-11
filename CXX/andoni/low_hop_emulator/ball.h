#ifndef __BALL_H__
#define __BALL_H__

#include <ctf.hpp>
#include <float.h>
#include <math.h>

using namespace CTF;
#define SEED 23
typedef float REAL;
#define MAX_REAL (INT_MAX/2)

#define BALL_SIZE 4

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

// Monoid<bpair> get_bpair_monoid();

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

namespace CTF {
  template<>
  inline void Set<bvector<BALL_SIZE>>::print(char const * a, FILE * fp) const {
    bvector<BALL_SIZE> * bvec = (bvector<BALL_SIZE>*)a;
    for (int i = 0; i < BALL_SIZE; ++i) {
      fprintf(fp, "(%d %d)", bvec->closest_neighbors[i].col, bvec->closest_neighbors[i].dist);
    }
  }
}

template<int b>
void bvector_red(bvector<b> const * x, // TODO: do n times
                 bvector<b> * y,
                 int n){
  std::set<int> cols;
  int i,j,k;
  i = j = k = 0;
  while (k < b && i < b && j < b) { // TODO: write as parallelizable for loop
    if (x->closest_neighbors[i].col == -1) {
      ++j;
      ++k;
    } else if (y->closest_neighbors[j].col == -1) {
      y->closest_neighbors[k].col = x->closest_neighbors[i].col;
      y->closest_neighbors[k].dist = x->closest_neighbors[i].dist;
      ++i;
      ++k;
    }  else if (x->closest_neighbors[i].dist < y->closest_neighbors[j].dist) {
      if (cols.find(x->closest_neighbors[i].col) != cols.end()) { cols.insert(x->closest_neighbors[i].col); }
      else { ++i; continue; }
      y->closest_neighbors[k].col = x->closest_neighbors[i].col;
      y->closest_neighbors[k].dist = x->closest_neighbors[i].dist;
      ++i;
      ++k;
    } else {
      if (cols.find(y->closest_neighbors[j].col) != cols.end()) { cols.insert(y->closest_neighbors[j].col); }
      else { ++j; continue; }
      ++j;
      ++k;
    }
  }
  if (j == b) {
    while (k < b) {
      if (cols.find(x->closest_neighbors[i].col) != cols.end()) { cols.insert(x->closest_neighbors[i].col); }
      else { ++i; continue; }
      y->closest_neighbors[k].col = x->closest_neighbors[i].col;
      y->closest_neighbors[k].dist = x->closest_neighbors[i].dist;
      ++i;
      ++k;
    }
  }
  std::sort(y->closest_neighbors, y->closest_neighbors + b,
            [](bpair const & first, bpair const & second) -> bool
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
          while (k < b && i < b && j < b) { // TODO: write as parallelizable for loop
            if (x.closest_neighbors[i].col == -1) {
              z.closest_neighbors[k].col = y.closest_neighbors[j].col;
              z.closest_neighbors[k].dist = y.closest_neighbors[j].dist;
              ++j;
              ++k;
            } else if (y.closest_neighbors[j].col == -1) {
              z.closest_neighbors[k].col = x.closest_neighbors[i].col;
              z.closest_neighbors[k].dist = x.closest_neighbors[i].dist;
              ++i;
              ++k;
            } else if (x.closest_neighbors[i].dist < y.closest_neighbors[j].dist) {
              if (cols.find(x.closest_neighbors[i].col) != cols.end()) { cols.insert(x.closest_neighbors[i].col); }
              else { ++i; continue; }
              z.closest_neighbors[k].col = x.closest_neighbors[i].col;
              z.closest_neighbors[k].dist = x.closest_neighbors[i].dist;
              ++i;
              ++k;
            } else {
              if (cols.find(y.closest_neighbors[j].col) != cols.end()) { cols.insert(y.closest_neighbors[j].col); }
              else { ++j; continue; }
              z.closest_neighbors[k].col = y.closest_neighbors[j].col;
              z.closest_neighbors[k].dist = y.closest_neighbors[j].dist;
              ++j;
              ++k;
            }
          }
          if (i == b) {
            while (k < b) {
              assert(j < b); // FIXME: TODO: at first, every col is -1
              // if (cols.find(y.closest_neighbors[j].col) != cols.end()) { cols.insert(y.closest_neighbors[j].col); }
              // else { ++j; continue; }
              z.closest_neighbors[k].col = y.closest_neighbors[j].col;
              z.closest_neighbors[k].dist = y.closest_neighbors[j].dist;
              ++j;
              ++k;
            }
          } else if (j == b) {
            while (k < b) {
              assert(i < b);
              // if (cols.find(x.closest_neighbors[i].col) != cols.end()) { cols.insert(x.closest_neighbors[i].col); }
              // else { ++i; continue; }
              z.closest_neighbors[k].col = x.closest_neighbors[i].col;
              z.closest_neighbors[k].dist = x.closest_neighbors[i].dist;
              ++i;
              ++k;
            }
          }
          std::sort(z.closest_neighbors, z.closest_neighbors + b,
                    [](bpair const & first, bpair const & second) -> bool
                      { return first.dist < second.dist; }
                    ); // TODO: sort on insert?
          return z;
      },
      obvector
      );

  return m;
}

Matrix<REAL> * ball_matmat(Matrix<REAL> * A, int64_t b);

template<int b>
Vector<bvector<b>> * ball_bvector(Matrix<REAL> * A) {
  int n = A->nrow;
  World * w = A->wrld;
  Monoid<bvector<b>> bvector_monoid = get_bvector_monoid<b>();
  Vector<bvector<b>> * B = new Vector<bvector<b>>(n,*w,bvector_monoid);
  // int64_t A_npairs;
  // Pair<int> * A_loc_pairs;
  // A->get_local_pairs(&A_npair, &A_loc_pairs, true);
  // for (int64_t i = 0; i < A_npairs; ++i) { // collect smallest b edges for each vertex on processor
  //     
  // }
  // // reduce smallest b edges on each so each pair ha


  // Transform<REAL,bvector<b>> relax([](REAL w, bvector<b> bvec){
  Bivar_Function<REAL,bvector<b>,bvector<b>> relax([](REAL w, bvector<b> bvec){
    bvector<b> ret = bvec;
    for (int i = 0; i < b; ++i) {
      if (ret.closest_neighbors[i].dist == MAX_REAL) {
        ret.closest_neighbors[i].dist = w;
      } else {
        ret.closest_neighbors[i].dist += w;
      }
    }
    return ret;
  });

  for (int i = 0; i < b; ++i) {
    (*B)["i"] += relax((*A)["ij"], (*B)["j"]);
  }
  return B;
}

#endif
