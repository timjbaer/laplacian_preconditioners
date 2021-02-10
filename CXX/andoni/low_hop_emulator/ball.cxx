#include "ball.h"

/***** utility *****/
void write_valid_idxs(Matrix<REAL> * A, Pair<REAL> * pairs, int64_t npairs) {
  Pair<REAL> wr_pairs[npairs];
  int64_t nwrite = 0;
  for (int64_t i = 0; i < npairs; ++i) {
    if (pairs[i].k != -1) {
      wr_pairs[nwrite] = pairs[i];
      ++nwrite;
    }
  }
  A->write(nwrite, wr_pairs);
}

void write_first_b(Matrix<REAL> * A, Pair<REAL> * pairs, int64_t npairs, int b) {
  int n = A->nrow;
  Pair<REAL> wr_pairs[npairs];
  int64_t nwrite = 0;
  int64_t i = 0;
  while (i < npairs) {
    int vertex = pairs[i].k / n;
    int j = i;
    for (; j < i + b && j < npairs && pairs[j].k / n == vertex; ++j) {
      wr_pairs[nwrite] = pairs[j];
      ++nwrite;
      assert(nwrite < npairs);
    }
    while (j < npairs && pairs[j].k / n == vertex) {
      ++j;
    }
    i = j;
  }
  A->write(nwrite, wr_pairs);
}

/***** filter b closest neighbors *****/
void filter(Matrix<REAL> * A, int b) {
  int n = A->nrow; 
  int64_t A_npairs;
  Pair<REAL> * A_pairs;
  A->get_local_pairs(&A_npairs, &A_pairs, true); // FIXME: are get_local_pairs in sorted order by key?
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
                  [](Pair<REAL> const & first, Pair<REAL> const & second) -> bool
                    { return first.d < second.d; }
                    );
    // std::partial_sort(A_pairs + i*n, A_pairs + i*n + b, A_pairs + (i+1)*n, 
    //               [](Pair<bpair> const & first, Pair<bpair> const & second) -> bool
    //                 { return first.d.dist < second.d.dist; }
    //                 );
  }
  // if (A->wrld->rank == 0) {
  //   printf("HERE\n");
  //   for (int i = 0; i < A_npairs; ++i) {
  //     printf("(%d %f)\n", A_pairs[i].k, A_pairs[i].d);
  //   }
  // }
  // exit(1);
  (*A)["ij"] = MAX_REAL;
  A->sparsify();
  write_first_b(A, A_pairs, A_npairs, b);
  delete [] A_pairs;
}

/***** matmat approach *****/
Matrix<REAL> * ball_matmat(Matrix<REAL> * A, int64_t b) { // A should be on (min, +) semiring
  int n = A->nrow;
  int symm = A->symm;
  World wrld = *(A->wrld);
  Matrix<REAL> * B = new Matrix<REAL>(*A);
  for (int i = 0; i < log2(B->nrow); ++i) {
    (*B)["ij"] += (*B)["ik"] * (*B)["kj"];
    filter(B, b);
  }
  return B;
}

/***** matvec approach *****/
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
