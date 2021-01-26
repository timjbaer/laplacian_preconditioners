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

/***** filter b closest neighbors *****/
struct ball_t {
  int n;
  int b;
  Pair<REAL> closest_neighbors[]; // contiguous, flattened 2D array
};

void init_mpi(int n, int b);
void destroy_mpi();

ball_t * filter(Matrix<REAL> * A, int b);

class bpair {
  public:
    int vertex;
    REAL dist;

    bpair() { vertex = -1; dist = MAX_REAL; } // addid
    bpair(int vertex_, REAL dist_) { vertex = vertex_; dist = dist_; }
    bpair(bpair const & other) { vertex = other.vertex; dist = other.dist; }
};
Monoid<bpair> get_bpair_monoid();

ball_t * filter(Matrix<bpair> * A, int b);

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
  inline void Set<bvector<BALL_SIZE>>::print(char const * a, FILE * fp) const {
    bvector<BALL_SIZE> * bvec = (bvector<BALL_SIZE>*)a;
    for (int i = 0; i < BALL_SIZE; ++i) {
      fprintf(fp, "(%d %f)", bvec->closest_neighbors[i].vertex, bvec->closest_neighbors[i].dist);
    }
  }
}

template<int b>
// void bvector_red(bvector<b> const * x, // TODO: multithread
bvector<b> bvector_red(bvector<b> const * x, // TODO: multithread
                 bvector<b> * y,
                 int nitems){
  assert(nitems == 1); // TODO: correct?
  bvector<b> * y_prev = (bvector<b> *) malloc(sizeof(bvector<b>));
  for (int i = 0; i < b; ++i) {
    y_prev->closest_neighbors[i] = y->closest_neighbors[i];
  }
  std::set<int> seen;
  int i = 0;
  int j = 0;
  int k = 0;
  while (i < b && j < b) {
    if (x->closest_neighbors[i].vertex == -1 || y_prev->closest_neighbors[j].vertex == -1) {
      break;
    }
    if (x->closest_neighbors[i].dist < y_prev->closest_neighbors[j].dist) {
      int vertex = x->closest_neighbors[i].vertex;
      if (seen.find(vertex) == seen.end()) {
        seen.insert(vertex);
        y->closest_neighbors[k] = x->closest_neighbors[i];
        ++k;
      }
      ++i;
    } else {
      int vertex = y_prev->closest_neighbors[j].vertex;
      if (seen.find(vertex) == seen.end()) {
        seen.insert(vertex);
        y->closest_neighbors[k] = y_prev->closest_neighbors[j];
        ++k;
      }
      ++j;
    }
  }
  if (i == b || x->closest_neighbors[i].vertex == -1) {
    while (k < b) {
      int vertex = y_prev->closest_neighbors[j].vertex;
      if (vertex == -1 || seen.find(vertex) == seen.end()) {
        seen.insert(vertex);
        y->closest_neighbors[k] = y_prev->closest_neighbors[j];
        ++k;
      }
      ++j;
    }
  } else if (j == b || y_prev->closest_neighbors[j].vertex == -1) {
    while (k < b) {
      int vertex = x->closest_neighbors[i].vertex;
      if (vertex == -1 || seen.find(vertex) == seen.end()) {
        seen.insert(vertex);
        y->closest_neighbors[k] = x->closest_neighbors[i];
        ++k;
      }
      ++i;
    }
  }
  free(y_prev);
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
        // return bvector<b>();
        // return x;
        // return y;
        return bvector_red<b>(&x, &y, 1);
      },
      obvector
      );

  return m;
}

template<int b>
void init_closest_edges(Matrix<bpair> * A, Vector<bvector<b>> * B) {
  int n = A->nrow;
  ball_t * ball = filter(A, b);
  // printf("ball:\n");
  // for (int i = 0; i < n; ++i) {
  //   for (int j = 0; j < b; ++j) {
  //     printf("(%d %f) ", ball->closest_neighbors[i*b + j].k, ball->closest_neighbors[i*b + j].d);
  //   }
  //   printf("\n");
  // }
  Pair<bvector<b>> bvecs[n];
#ifdef _OPENMP
  #pragma omp parallel for
#endif
  for (int i = 0; i < n; ++i) {
    bvecs[i].k = i;
    for (int j = 0; j < b; ++j) {
      bvecs[i].d.closest_neighbors[j].vertex = -1;
      bvecs[i].d.closest_neighbors[j].dist = MAX_REAL;
    }
  }
  int off[n];
  memset(off, 0, n*sizeof(int));
  for (int i = 0; i < n*b; ++i) { // fills bvecs in sorted order
    Pair<REAL> pair = ball->closest_neighbors[i];
    int vertex = pair.k % n;
    if (pair.k != -1) {
      bvecs[vertex].d.closest_neighbors[off[vertex]].vertex = pair.k / n;
      bvecs[vertex].d.closest_neighbors[off[vertex]].dist = pair.d;
      ++off[vertex];
      assert(off[vertex] <= b);
    }
  }
  B->write(n, bvecs); 

  free(ball);
}

template<int b>
Vector<bvector<b>> * ball_bvector(Matrix<bpair> * A) {
  int n = A->nrow;
  World * w = A->wrld;
  Monoid<bvector<b>> bvector_monoid = get_bvector_monoid<b>();
  Vector<bvector<b>> * B = new Vector<bvector<b>>(n, *w, bvector_monoid);
  init_closest_edges(A, B);
  printf("B with closest edges\n");
  B->print();

  Bivar_Function<bpair,bvector<b>,bvector<b>> relax([](bpair e, bvector<b> bvec){ // TODO: use Transform (as long as it accumulates)
    bvector<b> ret = bvec;
    for (int i = 0; i < b; ++i) {
      if (ret.closest_neighbors[i].vertex == -1) {
        ret.closest_neighbors[i] = bvec.closest_neighbors[i];
      } else {
        ret.closest_neighbors[i].vertex = bvec.closest_neighbors[i].vertex;
        ret.closest_neighbors[i].dist = e.dist + bvec.closest_neighbors[i].dist;
      }
    }
    return ret;
  });

  for (int i = 0; i < b; ++i) {
  // for (int i = 0; i < 1; ++i) {
    (*B)["i"] += relax((*A)["ij"], (*B)["j"]);
  }

  return B;
}

#endif
