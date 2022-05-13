#ifndef __BALL_H__
#define __BALL_H__

#include "shared.h"

// ==================================================================
// b closest neighbor set
// ==================================================================
// a b closest neighbor set (referred here as "ball") is a set that
// is centered at some vertex and contains its b closest vertices
// ==================================================================

#define BALL_SIZE 4 // b

// common -----------------------------------------------------------

/* the base unit in a closest neighbor set
 * a bpair is a tuple that contains a vertex id v, and then d(v,c),
 * the distance from the vertex at the center of the ball c to v
 */
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


/* a data structure for a closest neighbor set
 * bvector is a vector of bpairs that represent the b closest
 * neighor vertices to the center of the set
 *
 * the center has a bpair with value 0 and is at closest_neighbor[0]
 *
 */
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

// print functions --------------------------------------------------
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

// algebraic functions and reductions for CTF  ----------------------

static Semiring<REAL> MIN_PLUS_SR(MAX_REAL,
    [](REAL a, REAL b) {
      return std::min(a, b);
    },
    MPI_MIN,
    0,
    [](REAL a, REAL b) {
      return a + b;
    });

Monoid<bpair> get_bpair_monoid();


/* creates an MPI reduction over two sets of b-closest neighbor sets
 * the reduction will, between two different bCN centered at some v, y[i]
 * and x[i], find the new b-closest neighbors to v between the two
 *
 * input: an array x of bvectors x[i]
 *        an array y of bvectors y[i]
 *        nitems is the length of x and y
 *
 * output: array of bvectors that contains the bCN between x[i] and y[i]
 */
template<int b>
bvector<b> bvector_red(bvector<b> const * x,
                 bvector<b> * y,
                 int nitems){
  Timer t_bvector_red("bvector_red");
  t_bvector_red.start();

#ifdef _OPENMP
  #pragma omp parallel for
#endif
  for (int item = 0; item < nitems; ++item) {
    if (x[item].closest_neighbors[0].vertex == -1) {
      // if there are no vertices in ball x[i]
      continue;
    } else if (y[item].closest_neighbors[0].vertex == -1) {
      // if there are no vertices in ball y[i]
      y[item] = x[item];
      continue;
    }
    if (x[item].closest_neighbors[0].dist >= y[item].closest_neighbors[b-1].dist) {
      // none of the neighbors in x[i] are closer than those already in y[i]
      // should be taken much more often than below elseif due to reordering
      continue;
    } else if (y[item].closest_neighbors[0].dist >= x[item].closest_neighbors[b-1].dist) {
      // all of the neighbors in x[i] are closer than those already in y[i]
      y[item] = x[item];
      continue;
    }

    // add contents of the x[i] and y[i] that we currently work on to result
    int nnz = 0;
    bvector<2*b> res;
    for (int i = 0; i < b; ++i) {
      if (y[item].closest_neighbors[i].vertex == -1)
        break;
      res.closest_neighbors[nnz] = y[item].closest_neighbors[i];
      ++nnz;
    }
    for (int i = 0; i < b; ++i) {
      if (x[item].closest_neighbors[i].vertex == -1)
        break;
      res.closest_neighbors[nnz] = x[item].closest_neighbors[i];
      ++nnz;
    }

    // sort result by vertex index
    std::sort(res.closest_neighbors, res.closest_neighbors + nnz, 
                  [](bpair const & first, bpair const & second) -> bool
                    { 
                      if (first.vertex != second.vertex)
                        return first.vertex < second.vertex;
                      else
                        return first.dist < second.dist; // break ties by distance
                    });

    // remove duplicate vertices in result
    int prev = res.closest_neighbors[0].vertex; // assumes isolated vertices removed
    for (int i = 1; i < nnz; ++i) {
      int vertex = res.closest_neighbors[i].vertex;
      if (vertex == prev) {
        res.closest_neighbors[i].vertex = -1;
        res.closest_neighbors[i].dist = MAX_REAL;
      } else {
        prev = vertex;
      }
    }

    // sort result by distance
    std::partial_sort(res.closest_neighbors, res.closest_neighbors + std::min(b,nnz), res.closest_neighbors + nnz,
                  [](bpair const & first, bpair const & second) -> bool
                    { return first.dist < second.dist; }
                    );

    // this updates y[i] with the new bCN vertices from (min_b)(y[i], x[i])
    for (int i = 0; i < std::min(b,nnz); ++i) {
      y[item].closest_neighbors[i] = res.closest_neighbors[i];
    }
    for (int i = nnz; i < b; ++i) {
      y[item].closest_neighbors[i].vertex = -1;
      y[item].closest_neighbors[i].dist = MAX_REAL;
    }
  }

  t_bvector_red.stop();
  return *y; // result only used when nitems == 1
}

// creates a monoid using CTF using bvector_red used in ball_bvector
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

// utility functions ------------------------------------------------

void write_first_b(Matrix<REAL> * A, Pair<REAL> * pairs, int64_t npairs);

// filter shortest paths matrix A for the b-closest values
void filter(Matrix<REAL> * A, int b);


template<int b>
void init_closest_edges(Matrix<REAL> * A, Vector<bvector<b>> * B) {
  Timer t_init_closest_edges("init_closest_edges");
  t_init_closest_edges.start();

  int n = A->nrow; 
  World * w = A->wrld;
  Vector<int> v = arange<int>(0, n, 1, *w); // create vector of vertex indices

  // instantiate each bCN centered at i to have the bpair (i, 0)
  Transform<int,bvector<b>>(
          [](int i, bvector<b> & bvec){
            bvec.closest_neighbors[0].vertex = i;
            bvec.closest_neighbors[0].dist = 0.0; // a vertex is one of its closest neighbors
          }
        )(v["i"], (*B)["i"]);

  Bivar_Function<REAL,int,bvector<b>> f(
          [](REAL a, int j){
            bvector<b> bvec;
            bvec.closest_neighbors[0] = bpair(j, a);
            return bvec;
          }
        );

  f.intersect_only = true;

  // adds all the elements in the adjacency matrix A to vector of bvectors B
  // (*B)["i"] = f((*A)["ij"], v["j"]);
  (*B)["i"] += f((*A)["ij"], v["j"]);
  t_init_closest_edges.stop();
}

template<int b>
int64_t are_vectors_different(Vector<bvector<b>> * A, Vector<bvector<b>> * B)
{
  Scalar<int64_t> s;
  s[""] += Function<bvector<b>,bvector<b>,int64_t>(
             [](bvector<b> a, bvector<b> c){
               return (int64_t) fabs(a!=c);
             }
           )((*A)["i"],(*B)["i"]);
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
  delete [] B_pairs;
}

// ball computation functions  --------------------------------------
// move *

Matrix<REAL> * ball_matmat(Matrix<REAL> * A, int b);

template<int b>
Vector<bvector<b>> * ball_bvector(Matrix<REAL> * A, int conv, int square) { // see section 3.1 & 3.2
  assert(A->is_sparse); // not strictly necessary, but much more efficient
  int n = A->nrow;
  World * w = A->wrld;
  Monoid<bvector<b>> bvector_monoid = get_bvector_monoid<b>();
  Vector<bvector<b>> * B = new Vector<bvector<b>>(n, *w, bvector_monoid);
  init_closest_edges(A, B);

  Bivar_Function<REAL,bvector<b>,bvector<b>> relax([](REAL a, bvector<b> bvec){ // TODO: use Transform (as long as it accumulates)
    assert(fabs(MAX_REAL - a) >= EPSILON); // since intersect_only = true
    for (int i = 0; i < b; ++i) {
      if (bvec.closest_neighbors[i].vertex > -1)
        bvec.closest_neighbors[i].dist += a;
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
      printf("converged in %d iterations (do not count iteration where B == B_prev)\n", niter - 1);
    delete B_prev;
  } else {
    if (square) {
      for (int i = 0; i < log2(b); ++i) {
        t_relax.start();
        (*B)["i"] += relax((*A)["ij"], (*B)["j"]);
        t_relax.stop();
        t_square.start();
        bvec_to_mat(A, B); // writes B to A
        t_square.stop();
      }
    } else {
      t_relax.start();
      for (int i = 0; i < b; ++i) {
        (*B)["i"] += relax((*A)["ij"], (*B)["j"]);
      }
      t_relax.stop();
    }
  }

  return B;
}

template<int b>
Vector<bvector<b>> * ball_multilinear(Matrix<REAL> * A, int conv, int square) { // see section 3.4 & 3.5
  assert(A->is_sparse); // not strictly necessary, but much more efficient
  int n = A->nrow;
  World * w = A->wrld;
  Monoid<bvector<b>> bvector_monoid = get_bvector_monoid<b>();
  Vector<bvector<b>> * B = new Vector<bvector<b>>(n, *w, bvector_monoid);
  init_closest_edges<b>(A, B);

  std::function<bvector<b>(bvector<b>,REAL,bvector<b>)> f = [](bvector<b> other, REAL a, bvector<b> me){ // me and other are switched since CTF stores col-first
    assert(fabs(MAX_REAL - a) >= EPSILON);
    if (other.closest_neighbors[0].vertex == -1 || other.closest_neighbors[0].dist + a >= me.closest_neighbors[b-1].dist)
      return bvector<b>(); // me will be accumulated into
    for (int i = 0; i < b; ++i) {
      if (other.closest_neighbors[i].vertex > -1)
        other.closest_neighbors[i].dist += a;
    }
    // comment out below 2 lines and uncomment following line to recover bvector algorithm (with additional overhead)
    bvector_red<b>(&other, &me, 1);
    return me;
    // return other;
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
      printf("converged in %d iterations (do not count iteration where B == B_prev)\n", niter - 1);
    delete B_prev;
  } else {
    if (square) {
      for (int i = 0; i < log2(b); ++i) {
        Multilinear<REAL,bvector<b>,bvector<b>>(A, vec_list, B, f);
        t_square.start();
        bvec_to_mat(A, B); // writes B to A
        t_square.stop();
      }
    } else {
      for (int i = 0; i < b; ++i) {
        Multilinear<REAL,bvector<b>,bvector<b>>(A, vec_list, B, f);
      }
    }
  }

  return B;
}

#endif
