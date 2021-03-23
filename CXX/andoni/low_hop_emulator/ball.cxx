#include "ball.h"

/***** utility *****/
void write_first_b(Matrix<REAL> * A, Pair<REAL> * pairs, int64_t npairs, int b) {
  int n = A->nrow;
  Pair<REAL> wr_pairs[npairs];
  int64_t nwrite = 0;
  int64_t i = 0;
  while (i < npairs) {
    int vertex = pairs[i].k % n;
    int64_t j = i;
    for (; j < i + b && j < npairs && pairs[j].k % n == vertex; ++j) {
      wr_pairs[nwrite] = pairs[j];
      ++nwrite;
      assert(nwrite <= npairs);
    }
    while (j < npairs && pairs[j].k % n == vertex) {
      ++j;
    }
    i = j;
  }
  A->write(nwrite, wr_pairs);
}

void filter(Matrix<REAL> * A, int b) {
  assert(A->topo->order == 1 || A->wrld->np == 1); // A distributed on 1D processor grid
  Timer t_filter("filter");
  t_filter.start();
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
  // (*A)["ij"] = MAX_REAL; // FIXME: why do we need to use set_zero?
  A->set_zero();
  A->sparsify();
  write_first_b(A, A_pairs, A_npairs, b);
  delete [] A_pairs;
  t_filter.stop();
}

/***** matmat *****/
Matrix<REAL> * ball_matmat(Matrix<REAL> * A, int b) { // A should be on (min, +) semiring
  assert(A->is_sparse); // not strictly necessary, but much more efficient
  int n = A->nrow;
  Matrix<REAL> * B = new Matrix<REAL>(*A);
  Timer t_matmat("matmat");
  for (int i = 0; i < log2(n); ++i) {
    t_matmat.start();
    (*B)["ij"] += (*B)["ik"] * (*B)["kj"]; // FIXME: is this a sparse contraction?
    t_matmat.stop();
    filter(B, b);
  }
  return B;
}

/***** common *****/
bpair bpair_min(bpair a, bpair b){
  return a.dist < b.dist ? a : b;
}

void bpair_red(bpair const * a,
               bpair * b,
               int n){
#ifdef _OPENMP
  #pragma omp parallel for
#endif
  for (int i=0; i<n; i++){
    b[i] = bpair_min(a[i], b[i]);
  } 
}

Monoid<bpair> get_bpair_monoid(){ // FIXME: causes "Attempting to use an MPI routine after finalizing MPICH" error
    MPI_Op omee;
    MPI_Op_create(
      [](void * a, void * b, int * n, MPI_Datatype*){ 
        bpair_red((bpair*)a, (bpair*)b, *n);
      },
    1, 
    &omee);

    Monoid<bpair> MIN_BPAIR(
      bpair(), 
      [](bpair a, bpair b){ return bpair_min(a, b); }, 
      omee);

  return MIN_BPAIR; 
}
