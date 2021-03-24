#include "test.h"
#include "../low_hop_emulator.h"

int main(int argc, char** argv)
{
  int const in_num = argc;
  char** input_str = argv;
  int critter_mode=0;
  int b=BALL_SIZE;
  int d=2;
  int mode=0;

  int rank;
  int np;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  {
    World w(argc, argv);
    Matrix<REAL> * B = get_graph(argc, argv, w);
    if (getCmdOption(input_str, input_str+in_num, "-d")){
      d = atoi(getCmdOption(input_str, input_str+in_num, "-d"));
      if (d < 1 || d > 2) d = 2;
    }
    if (getCmdOption(input_str, input_str+in_num, "-mode")){
      mode = atoi(getCmdOption(input_str, input_str+in_num, "-mode"));
      if (mode < 0 || mode > 1) mode = 0;
    }
    Matrix<REAL> * A;
    if (d == 1) { // 1D distribution (block along rows)
      int plens[1] = { np };
      Partition part(1, plens);
      Idx_Partition blk;
      A = new Matrix<REAL>(B->nrow, B->ncol, "ij", part["i"], blk, B->symm|(B->is_sparse*SP), w, MIN_PLUS_SR);
    } else { // default (2D) distribution
      A = new Matrix<REAL>(B->nrow, B->ncol, B->symm|(B->is_sparse*SP), w, MIN_PLUS_SR);
    }
    (*A)["ij"] = (*B)["ij"]; // change to (min, +) semiring and correct distribution
    assert(A->is_sparse); // not strictly necessary, but much more efficient
    delete B;
#ifdef CRITTER
      critter::start(critter_mode);
#endif
      if (A != NULL) {
        LowHopEmulator lhe(A, b, mode);
#ifdef DEBUG
        int64_t G_nnz = 0;
        if (mode) {
          if (w.rank == 0)
            printf("low hop emulator\n");
          lhe.G->print_matrix();
          G_nnz = lhe.G->nnz_tot;
        }
        Matrix<REAL> * A_dist = correct_dist(A, b);
        Matrix<REAL> * G_dist = lhe.sssp();
        Matrix<REAL> * D = new Matrix<REAL>(A->nrow, A->ncol, A->symm|(A->is_sparse*SP), w, PLUS_TIMES_SR);
        Bivar_Function<REAL,REAL,REAL> distortion([](REAL x, REAL y){ return y > 0 ? x / y : 1; });
        distortion.intersect_only = true;
        (*D)["ij"] = distortion((*G_dist)["ij"], (*A_dist)["ij"]);
        double max_distort = D->norm_infty();
        Scalar<REAL> tot_distort(w);
        tot_distort[""] = Function<REAL,REAL>([](REAL x){ return x; })((*D)["ij"]);
        double avg_distort = tot_distort.get_val() / D->nnz_tot;
        delete G_dist;
        delete A_dist;
        delete D;
        if (w.rank == 0) {
          if (mode)
            printf("low hop emulator has %" PRId64 " nonzeros\n", G_nnz);
          printf("low hop emulator max distance distortion between vertices is %f\n", max_distort);
          printf("low hop emulator avg distance distortion between vertices is %f\n", avg_distort);
        }
#endif
        delete A;
      }
#ifdef CRITTER
      critter::stop(critter_mode);
#endif
  }
  MPI_Finalize();
  return 0;
}
