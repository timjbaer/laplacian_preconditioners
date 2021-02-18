#ifndef __BALL_H__
#define __BALL_H__

#include <ctf.hpp>
#include <float.h>
#include <math.h>

using namespace CTF;
#define SEED 23
typedef float REAL;
#define MAX_REAL  (INT_MAX/2)
#define EPSILON   0.01
#define BALL_SIZE 8

class bpair {
  public:
    int vertex;
    REAL dist;

    bpair() { vertex = -1; dist = MAX_REAL; } // addid
    bpair(int vertex_, REAL dist_) { vertex = vertex_; dist = dist_; }
    bpair(bpair const & other) { vertex = other.vertex; dist = other.dist; }
};
Monoid<bpair> get_bpair_monoid();

/***** utility *****/
void write_valid_idxs(Matrix<REAL> * A, Pair<REAL> * pairs, int64_t npairs);
void write_first_b(Matrix<REAL> * A, Pair<REAL> * pairs, int64_t npairs);

/***** filter b closest neighbors *****/
void filter(Matrix<REAL> * A, int b);

/***** matmat approach *****/
static Semiring<REAL> MIN_PLUS_SR(MAX_REAL,
    [](REAL a, REAL b) {
      return std::min(a, b);
    },
    MPI_MIN,
    0,
    [](REAL a, REAL b) {
      return a + b;
    });

Matrix<REAL> * ball_matmat(Matrix<REAL> * A, int64_t b);

/***** matvec approach *****/
template<int b>
class bvector {
  public: 
    bpair closest_neighbors[b];

    bvector() { 
      for (int i = 0; i < b; ++i) {
        closest_neighbors[i].vertex = -1;
        closest_neighbors[i].dist = MAX_REAL;
      }
    }
};

namespace CTF {
  template<>
  inline void Set<bpair>::print(char const * a, FILE * fp) const {
      bpair * b = (bpair*)a;
      fprintf(fp, "(%d %f)", b->vertex, b->dist);
  }

  template<>
  inline void Set<bvector<BALL_SIZE>>::print(char const * a, FILE * fp) const {
    bvector<BALL_SIZE> * bvec = (bvector<BALL_SIZE>*)a;
    for (int i = 0; i < BALL_SIZE; ++i) {
      fprintf(fp, "(%d %f)", bvec->closest_neighbors[i].vertex, bvec->closest_neighbors[i].dist);
    }
  }
}

template<int b>
bvector<b> bvector_red(bvector<b> const * x, // TODO: multithread
                 bvector<b> * y,
                 int nitems){
Timer t_bvector_red("bvector_red");
t_bvector_red.start();
#ifdef _OPENMP
  #pragma omp parallel for
#endif
  for (int item = 0; item < nitems; ++item) {
    if (x->closest_neighbors[0].dist >= y->closest_neighbors[b-1].dist)
      continue;
    bvector<b> * y_prev = (bvector<b> *) malloc(sizeof(bvector<b>));
    for (int i = 0; i < b; ++i) {
      y_prev->closest_neighbors[i] = y[item].closest_neighbors[i];
    }
    std::set<int> seen;
    int i = 0;
    int j = 0;
    int k = 0;
    while (i < b && j < b && k < b) {
      if (x[item].closest_neighbors[i].vertex == -1 || y_prev->closest_neighbors[j].vertex == -1) {
        break;
      }
      if (x[item].closest_neighbors[i].dist < y_prev->closest_neighbors[j].dist) {
        int vertex = x[item].closest_neighbors[i].vertex;
        if (seen.find(vertex) == seen.end()) {
          seen.insert(vertex);
          y[item].closest_neighbors[k] = x[item].closest_neighbors[i];
          ++k;
        }
        ++i;
      } else {
        int vertex = y_prev->closest_neighbors[j].vertex;
        if (seen.find(vertex) == seen.end()) {
          seen.insert(vertex);
          y[item].closest_neighbors[k] = y_prev->closest_neighbors[j];
          ++k;
        }
        ++j;
      }
    }
    if (i == b || x[item].closest_neighbors[i].vertex == -1) {
      while (k < b) {
        int vertex = y_prev->closest_neighbors[j].vertex;
        if (vertex == -1 || seen.find(vertex) == seen.end()) {
          seen.insert(vertex);
          y[item].closest_neighbors[k] = y_prev->closest_neighbors[j];
          ++k;
        }
        ++j;
      }
    } else if (j == b || y_prev->closest_neighbors[j].vertex == -1) {
      while (k < b) {
        int vertex = x[item].closest_neighbors[i].vertex;
        if (vertex == -1 || seen.find(vertex) == seen.end()) {
          seen.insert(vertex);
          y[item].closest_neighbors[k] = x[item].closest_neighbors[i];
          ++k;
        }
        ++i;
      }
    }
    free(y_prev);
  }
  t_bvector_red.stop();
  return *y;
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
        return bvector_red<b>(&x, &y, 1);
      },
      obvector
      );

  return m;
}

// template<int b>
// void init_closest_edges(Matrix<bpair> * A, Vector<bvector<b>> * B) {
//   int n = A->nrow; 
//   int64_t A_npairs;
//   Pair<bpair> * A_pairs;
//   A->get_local_pairs(&A_npairs, &A_pairs, true);
//   std::sort(A_pairs, A_pairs + A_npairs,
//         std::bind([](Pair<bpair> const & first, Pair<bpair> const & second, int64_t n) -> bool
//                     { return first.k % n < second.k % n; }, 
//                     std::placeholders::_1, std::placeholders::_2, n)
//            );
// 
//   int np = A->wrld->np;
//   int64_t off[(int)ceil(n/(float)np)+1];
//   int vertex = -1;
//   int nrows = 0;
//   for (int i = 0; i < A_npairs; ++i) {
//     if (A_pairs[i].k % n > vertex) {
//       off[nrows] = i;
//       vertex = A_pairs[i].k % n;
//       ++nrows;
//     }
//   }
//   off[nrows] = A_npairs;
// #ifdef _OPENMP
//   #pragma omp parallel for
// #endif
//   for (int64_t i = 0; i < nrows; ++i) { // sort to filter b closest edges
//     int nedges = off[i+1] - off[i];
//     int64_t first = off[i];
//     int64_t middle = off[i] + (nedges < b ? nedges : b);
//     int64_t last = off[i] + nedges;
//     std::partial_sort(A_pairs + first, A_pairs + middle, A_pairs + last, 
//                   [](Pair<bpair> const & first, Pair<bpair> const & second) -> bool
//                     { return first.d.dist < second.d.dist; }
//                     );
//   }
//   Pair<bvector<b>> bvecs[nrows];
// #ifdef _OPENMP
//   #pragma omp parallel for
// #endif
//   for (int i = 0; i < nrows; ++i) {
//     bvecs[i].k = A_pairs[off[i]].k % n;
//     int nedges = off[i+1] - off[i];
//     for (int j = 0; j < (nedges < b ? nedges : b); ++j) {
//       bvecs[i].d.closest_neighbors[j] = A_pairs[off[i] + j].d;
//     }
//   }
//   B->write(nrows, bvecs); 
//   delete [] A_pairs;
// }

template<int b>
void init_closest_edges(Matrix<REAL> * A, Vector<bvector<b>> * B) {
  Timer t_init_closest_edges("init_closest_edges");
  t_init_closest_edges.start();
  int n = A->nrow; 
  int64_t A_npairs;
  Pair<REAL> * A_pairs;
  A->get_local_pairs(&A_npairs, &A_pairs, true);
  std::sort(A_pairs, A_pairs + A_npairs,
        std::bind([](Pair<REAL> const & first, Pair<REAL> const & second, int64_t n) -> bool
                    { return first.k % n < second.k % n; }, 
                    std::placeholders::_1, std::placeholders::_2, n)
           );

  int np = A->wrld->np;
  int64_t off[(int)ceil(n/(float)np)+1];
  int vertex = -1;
  int nrows = 0;
  for (int i = 0; i < A_npairs; ++i) {
    if (A_pairs[i].k % n > vertex) {
      off[nrows] = i;
      vertex = A_pairs[i].k % n;
      ++nrows;
    }
  }
  off[nrows] = A_npairs;
#ifdef _OPENMP
  #pragma omp parallel for
#endif
  for (int64_t i = 0; i < nrows; ++i) { // sort to filter b closest edges
    int nedges = off[i+1] - off[i];
    int64_t first = off[i];
    int64_t middle = off[i] + (nedges < b ? nedges : b);
    int64_t last = off[i] + nedges;
    std::partial_sort(A_pairs + first, A_pairs + middle, A_pairs + last, 
                  [](Pair<REAL> const & first, Pair<REAL> const & second) -> bool
                    { return first.d < second.d; }
                    );
  }
  Pair<bvector<b>> bvecs[nrows];
#ifdef _OPENMP
  #pragma omp parallel for
#endif
  for (int i = 0; i < nrows; ++i) {
    bvecs[i].k = A_pairs[off[i]].k % n;
    int nedges = off[i+1] - off[i];
    for (int j = 0; j < (nedges < b ? nedges : b); ++j) {
      bvecs[i].d.closest_neighbors[j].vertex = A_pairs[off[i] + j].k / n;
      bvecs[i].d.closest_neighbors[j].dist = A_pairs[off[i] + j].d;
    }
  }
  B->write(nrows, bvecs); 
  delete [] A_pairs;
  t_init_closest_edges.stop();
}

template<int b>
Vector<bvector<b>> * ball_bvector(Matrix<REAL> * A) {
  int n = A->nrow;
  World * w = A->wrld;
  Monoid<bvector<b>> bvector_monoid = get_bvector_monoid<b>();
  Vector<bvector<b>> * B = new Vector<bvector<b>>(n, *w, bvector_monoid);
  init_closest_edges(A, B);

  Bivar_Function<REAL,bvector<b>,bvector<b>> relax([](REAL a, bvector<b> bvec){ // TODO: use Transform (as long as it accumulates)
    assert(fabs(MAX_REAL - a) >= EPSILON); // since intersect_only = true
    for (int i = 0; i < b; ++i) {
      if (bvec.closest_neighbors[i].vertex > -1)
        bvec.closest_neighbors[i].dist = a + bvec.closest_neighbors[i].dist;
    }
    return bvec;
  });
  relax.intersect_only = true;

  Timer t_relax("relax");
  t_relax.start();
  for (int i = 0; i < b; ++i) {
    (*B)["i"] += relax((*A)["ij"], (*B)["j"]);
  }
  t_relax.stop();

  return B;
}

#endif
