#include "ball.h"

/***** utility *****/
void write_valid_idxs(Matrix<REAL> * A, Pair<REAL> * pairs, int npairs) {
  Pair<REAL> wr_pairs[npairs];
  int nwrite = 0;
  for (int i = 0; i < npairs; ++i) {
    if (pairs[i].k != -1) {
      wr_pairs[nwrite] = pairs[i];
      ++nwrite;
    }
  }
  A->write(nwrite, wr_pairs);
}

/***** filter b closest neighbors *****/
// TODO: add function (without sorting) to convert Matrix<REAL> * to ball_t *
void filter(Matrix<REAL> * A, int b) {
//   // assert(A->symm != SY);
//   // assert(A->edge_map[0].type == CTF_int::PHYSICAL_MAP);
//   int n = A->nrow;
//   int64_t A_npairs;
//   Pair<REAL> * A_loc_pairs;
//   A->get_local_pairs(&A_npairs, &A_loc_pairs, true);
//   ball_t * ball = (ball_t *) malloc(sizeof(ball_t) + n * b * sizeof(Pair<REAL>));
//   ball->n = n;
//   ball->b = b;
//   for (int64_t i = 0; i < n; ++i) {
//     for (int64_t j = 0; j < b; ++j) {
//       ball->closest_neighbors[i*b + j].k = -1;
//       ball->closest_neighbors[i*b + j].d = MAX_REAL;
//     }
//   }
//   for (int64_t i = 0; i < A_npairs; ++i) {
//     int vertex = A_loc_pairs[i].k % n;
//     REAL dist = A_loc_pairs[i].d;
//     if (dist < ball->closest_neighbors[vertex*b + b-1].d) { // TODO: not efficient
//       ball->closest_neighbors[vertex*b + b-1] = A_loc_pairs[i];
//       std::sort(ball->closest_neighbors + vertex*b, ball->closest_neighbors + vertex*b + b,
//                 [](Pair<REAL> const & first, Pair<REAL> const & second) -> bool
//                   { return first.d < second.d; }
//                 );
//     }
//   }
// 
//   // CTF_int::CommData row_commdata = A->topo->dim_comm[A->edge_map[0].cdt];
//   // row_commdata.activate(MPI_COMM_WORLD);
//   // MPI_Comm row_comm = row_commdata.cm;
//   // MPI_Allreduce(MPI_IN_PLACE, ball, 1, MPI_BALL, oball, row_comm);
//   // row_commdata.allred(MPI_IN_PLACE, ball, 1, MPI_BALL, oball);
// 
//   int rank;
//   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//   char * idx = NULL;
//   Idx_Partition prl;
//   Idx_Partition blk;
//   A->get_distribution(&idx, prl, blk);
//   int color = 1; // TODO: communicate across rows
// 
//   MPI_Comm row_comm;
//   MPI_Comm_split(MPI_COMM_WORLD, color, rank, &row_comm);
//   MPI_Allreduce(MPI_IN_PLACE, ball, 1, MPI_BALL, oball, row_comm);
// 
//   delete [] A_loc_pairs;
// 
//   return ball;
}

/***** matmat approach *****/
Matrix<REAL> * ball_matmat(Matrix<REAL> * A, int64_t b) { // A should be on (min, +) semiring
//   int n = A->nrow;
//   int symm = A->symm;
//   World wrld = *(A->wrld);
//   Matrix<REAL> * B = new Matrix<REAL>(*A);
//   for (int i = 0; i < log2(B->nrow); ++i) { // TODO: clear B instead of reallocating it
//     (*B)["ij"] += (*B)["ik"] * (*B)["kj"];
//     filter(B, b);
//     delete B;
//     B = new Matrix<REAL>(n, n, symm, wrld, MIN_PLUS_SR);
//     // B->write(n*b, ball->closest_neighbors);
//     write_valid_idxs(B, ball->closest_neighbors, n*b);
//   }
// 
//   return B;
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
