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
#define BALL_SIZE 4

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
#ifdef _OPENMP
  #pragma omp parallel for
#endif
  for (int item = 0; item < nitems; ++item) {
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

template<int b>
void init_closest_edges(Matrix<bpair> * A, Vector<bvector<b>> * B) {
  int n = A->nrow; 
  int64_t A_npairs;
  Pair<bpair> * A_pairs;
  A->get_local_pairs(&A_npairs, &A_pairs, true); // FIXME: are get_local_pairs in sorted order by key?
  // if (A->wrld->rank == 0) {
  //   printf("A loc pairs on rank 0 in init_closest_edges\n");
  //   for (int i = 0; i < A_npairs; ++i) {
  //     printf("(%d %f)\n", A_pairs[i].k, A_pairs[i].d.dist);
  //   }
  // }
  // exit(1);
  int np = A->wrld->np;
  int64_t off[(int)ceil(n/(float)np)+1];
  int vertex = A_pairs[0].k / n;
  int nrows = 0;
  for (int i = 0; i < A_npairs; ++i) {
    if (A_pairs[i].k / n == vertex) {
      off[nrows] = i;
      vertex += np;
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
                  [](Pair<bpair> const & first, Pair<bpair> const & second) -> bool
                    { return first.d.dist < second.d.dist; }
                    );
    // std::partial_sort(A_pairs + i*n, A_pairs + i*n + b, A_pairs + (i+1)*n, 
    //               [](Pair<bpair> const & first, Pair<bpair> const & second) -> bool
    //                 { return first.d.dist < second.d.dist; }
    //                 );
  }
  // if (A->wrld->rank == 0) {
  //   printf("HERE\n");
  //   for (int i = 0; i < A_npairs; ++i) {
  //     printf("(%d %f)\n", A_pairs[i].k, A_pairs[i].d.dist);
  //   }
  // }
  // exit(1);
  Pair<bvector<b>> bvecs[nrows];
#ifdef _OPENMP
  #pragma omp parallel for
#endif
  for (int i = 0; i < nrows; ++i) {
    bvecs[i].k = A_pairs[off[i]].k / n;
    int nedges = off[i+1] - off[i];
    for (int j = 0; j < nedges; ++j) {
      bvecs[i].d.closest_neighbors[j] = A_pairs[off[i] + j].d;
    }
  }
  B->write(nrows, bvecs); 
  delete [] A_pairs;
}

template<int b>
Vector<bvector<b>> * ball_bvector(Matrix<bpair> * A) { // FIXME: (2,\inf) problem
  int n = A->nrow;
  World * w = A->wrld;
  Monoid<bvector<b>> bvector_monoid = get_bvector_monoid<b>();
  Vector<bvector<b>> * B = new Vector<bvector<b>>(n, *w, bvector_monoid);
  init_closest_edges(A, B);

  Bivar_Function<bpair,bvector<b>,bvector<b>> relax([](bpair e, bvector<b> bvec){ // TODO: use Transform (as long as it accumulates)
    // assert(e.vertex > -1 && e.dist < MAX_REAL); // since intersect_only = true
    bvector<b> ret;
    for (int i = 0; i < b; ++i) {
      if (bvec.closest_neighbors[i].vertex == -1) {
        ret.closest_neighbors[i] = bvec.closest_neighbors[i];
      } else {
        ret.closest_neighbors[i].vertex = bvec.closest_neighbors[i].vertex;
        ret.closest_neighbors[i].dist = e.dist + bvec.closest_neighbors[i].dist;
      }
    }
    return ret;
  });
  relax.intersect_only = true;

  for (int i = 0; i < b; ++i) {
    (*B)["i"] += relax((*A)["ij"], (*B)["j"]);
  }

  return B;
}

#endif
