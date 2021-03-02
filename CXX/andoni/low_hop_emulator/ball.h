#ifndef __BALL_H__
#define __BALL_H__

#include <ctf.hpp>
#include <float.h>
#include <math.h>

using namespace CTF;
typedef float REAL;
#define SEED 23
#define MAX_REAL  (INT_MAX/2)
#define EPSILON   0.01
#define BALL_SIZE 16

/***** utility *****/
void write_first_b(Matrix<REAL> * A, Pair<REAL> * pairs, int64_t npairs);
void filter(Matrix<REAL> * A, int b);

/***** matmat *****/
static Semiring<REAL> MIN_PLUS_SR(MAX_REAL,
    [](REAL a, REAL b) {
      return std::min(a, b);
    },
    MPI_MIN,
    0,
    [](REAL a, REAL b) {
      return a + b;
    });

Matrix<REAL> * ball_matmat(Matrix<REAL> * A, int b);

/***** common *****/
class bpair {
  public:
    int vertex;
    REAL dist;

    bpair() { vertex = -1; dist = MAX_REAL; } // addid
    bpair(int vertex_, REAL dist_) { vertex = vertex_; dist = dist_; }
    bpair(bpair const & other) { vertex = other.vertex; dist = other.dist; }
    bpair& operator=(bpair other) { vertex = other.vertex; dist = other.dist; return *this; }

    friend bool operator!=(const bpair & b1, const bpair & b2) {
      return b1.vertex != b2.vertex || fabs(b1.dist - b2.dist) >= EPSILON;
    }
};
Monoid<bpair> get_bpair_monoid();

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
    bvector(bvector<b> const & other) {
      for (int i = 0; i < b; ++i) {
        closest_neighbors[i] = other.closest_neighbors[i];
      }
    }
    bvector<b>& operator=(bvector<b> other) {
      for (int i = 0; i < b; ++i) {
        closest_neighbors[i] = other.closest_neighbors[i];
      }
      return *this;
    }

    friend bool operator!=(const bvector<b> & b1, const bvector<b> & b2) {
      for (int i = 0; i < b; ++i) {
        if (b1.closest_neighbors[i] != b2.closest_neighbors[i])
          return true;
      }
      return false;
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
bvector<b> bvector_red(bvector<b> const * x, // TODO: optimize merging of sorted lists
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

template<int b>
void init_closest_edges(Matrix<REAL> * A, Vector<bvector<b>> * B) { // TODO: refactor from filter
  assert(A->topo->order == 1); // A distributed on 1D processor grid
  Timer t_init_closest_edges("init_closest_edges");
  t_init_closest_edges.start();
  int n = A->nrow; 
  int64_t A_npairs;
  Pair<REAL> * A_pairs;
  A->get_local_pairs(&A_npairs, &A_pairs, true);
  std::sort(A_pairs, A_pairs + A_npairs,
        std::bind([](Pair<REAL> const & first, Pair<REAL> const & second, int n) -> bool
                    { return first.k % n < second.k % n; }, 
                    std::placeholders::_1, std::placeholders::_2, n)
           );

  int np = A->wrld->np;
  int64_t off[(int)ceil(n/(float)np)+1];
  int vertex = -1;
  int nrows = 0;
  for (int64_t i = 0; i < A_npairs; ++i) {
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
  for (int i = 0; i < nrows; ++i) { // sort to filter b closest edges
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
int64_t are_vectors_different(Vector<bvector<b>> * A, Vector<bvector<b>> * B)
{
  Scalar<int64_t> s;
  s[""] += Function<bvector<b>,bvector<b>,int64_t>([](bvector<b> a, bvector<b> c){ return (int64_t) fabs(a!=c); })((*A)["i"],(*B)["i"]);
  return s.get_val();
}

template<int b>
void bvec_to_mat(Matrix<REAL> * A, Vector<bvector<b>> * B) {
  int n = B->len;
  int64_t B_npairs;
  Pair<bvector<BALL_SIZE>> * B_pairs;
  B->get_local_pairs(&B_npairs, &B_pairs, true);
  int64_t A_npairs = 0;
  Pair<REAL> A_pairs[B_npairs*b];
  for (int64_t i = 0; i < B_npairs; ++i) {
    for (int j = 0; j < b; ++j) {
      if (B_pairs[i].d.closest_neighbors[j].vertex > -1) {
        A_pairs[A_npairs].k = B_pairs[i].k + B_pairs[i].d.closest_neighbors[j].vertex * n;
        A_pairs[A_npairs].d = B_pairs[i].d.closest_neighbors[j].dist;
        ++A_npairs;
      }
    }
  }
  (*A)["ij"] = MAX_REAL;
  A->write(A_npairs, A_pairs);
  A->sparsify();
}

/***** algorithms *****/
template<int b>
Vector<bvector<b>> * ball_bvector(Matrix<REAL> * A, int conv, int square) { // see section 3.1 & 3.2
  assert(A->topo->order == 1); // A distributed on 1D processor grid
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
  Timer t_square("square");
  if (conv) { // see section 3.7
    Timer t_conv("conv");
    Vector<bvector<b>> * B_prev = new Vector<bvector<b>>(*B);
    int niter = 0;
    int64_t diff = 1;
    while (diff) {
      t_relax.start();
      (*B)["i"] += relax((*A)["ij"], (*B)["j"]);
      t_relax.stop();
      if (square) {
        t_square.start();
        bvec_to_mat(A, B); // writes B to A
        t_square.stop();
      }
      t_conv.start();
      diff = are_vectors_different<BALL_SIZE>(B, B_prev);
      (*B_prev)["i"] = (*B)["i"];
      t_conv.stop();
      ++niter;
    }
    if (A->wrld->rank == 0)
      printf("converged in %d iterations\n", niter);
  } else {
    t_relax.start();
    for (int i = 0; i < b; ++i) {
      (*B)["i"] += relax((*A)["ij"], (*B)["j"]);
    }
    t_relax.stop();
  }

  return B;
}

template<int b>
Vector<bvector<b>> * ball_multilinear(Matrix<REAL> * A, int conv, int square) { // see section 3.4 & 3.5
  assert(A->topo->order == 2); // A distributed on 2D processor grid
  int n = A->nrow;
  World * w = A->wrld;
  Monoid<bvector<b>> bvector_monoid = get_bvector_monoid<b>();
  Vector<bvector<b>> * B = new Vector<bvector<b>>(n, *w, bvector_monoid);
  init_closest_edges(A, B);

  std::function<bvector<b>(bvector<b>,REAL,bvector<b>)> f = [](bvector<b> other, REAL a, bvector<b> me){ // me and other are switched since CTF stores col-first
    assert(fabs(MAX_REAL - a) >= EPSILON);
    if (other.closest_neighbors[0].dist + a >= me.closest_neighbors[b-1].dist)
      return bvector<b>(); // me will be accumulated into
    for (int i = 0; i < b; ++i) {
      if (other.closest_neighbors[i].vertex > -1)
        other.closest_neighbors[i].dist = a + other.closest_neighbors[i].dist;
    }
    // uncomment below line and comment out following 7 lines to recover bvector algorithm (with additional overhead)
    // return other;
    if (other.closest_neighbors[b-1].dist >= me.closest_neighbors[b-1].dist) {
      bvector_red<b>(&other, &me, 1); // write to better guess
      return me;
    } else {
      bvector_red<b>(&me, &other, 1);
      return other; // write to better guess
    }
  };
  Tensor<bvector<b>> * vec_list[2] = {B, B};

  Timer t_square("square");
  if (conv) { // see section 3.7
    Timer t_conv("conv");
    Vector<bvector<b>> * B_prev = new Vector<bvector<b>>(*B);
    int niter = 0;
    int64_t diff = 1;
    while (diff) {
      Multilinear<REAL,bvector<b>,bvector<b>>(A, vec_list, B, f);
      if (square) {
        t_square.start();
        bvec_to_mat(A, B); // writes B to A
        t_square.stop();
      }
      t_conv.start();
      diff = are_vectors_different<BALL_SIZE>(B, B_prev);
      (*B_prev)["i"] = (*B)["i"];
      t_conv.stop();
      ++niter;
    }
    if (A->wrld->rank == 0)
      printf("converged in %d iterations\n", niter);
  } else {
    for (int i = 0; i < b; ++i) {
      Multilinear<REAL,bvector<b>,bvector<b>>(A, vec_list, B, f);
    }
  }

  return B;
}

#endif
