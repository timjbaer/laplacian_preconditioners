#include "test.h"
#include "low_hop_emulator.h"

int main(int argc, char** argv)
{
  int const in_num = argc;
  char** input_str = argv;
  int critter_mode=0;
  int b=BALL_SIZE;

  int rank;
  int np;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  {
    World w(argc, argv);
    Matrix<REAL> * B = get_graph(argc, argv, w);
    // if (getCmdOption(input_str, input_str+in_num, "-d")){
    //   d = atoi(getCmdOption(input_str, input_str+in_num, "-d"));
    //   if (d < 1 || d > 2) d = 2;
    // }
    // Matrix<REAL> * A;
    // if (d == 1) { // 1D distribution (block along rows)
    //   int plens[1] = { np };
    //   Partition part(1, plens);
    //   Idx_Partition blk;
    //   A = new Matrix<REAL>(B->nrow, B->ncol, "ij", part["i"], blk, B->symm, w, MIN_PLUS_SR);
    // } else { // default (2D) distribution
    // Matrix<REAL> * A = new Matrix<REAL>(B->nrow, B->ncol, B->symm, w, MIN_PLUS_SR);
    Matrix<REAL> * A = new Matrix<REAL>(B->nrow, B->ncol, B->symm|(B->is_sparse*SP), w, MIN_PLUS_SR);
    // }
    (*A)["ij"] = (*B)["ij"]; // change to (min, +) semiring and correct distribution
    assert(A->is_sparse); // not strictly necessary, but much more efficient
    delete B;
    if (getCmdOption(input_str, input_str+in_num, "-b")){
      b = atoi(getCmdOption(input_str, input_str+in_num, "-b"));
      if (b < 0) b = 0;
    }
    if (getCmdOption(input_str, input_str+in_num, "-critter_mode")){
      critter_mode = atoi(getCmdOption(input_str, input_str+in_num, "-critter_mode"));
      if (critter_mode < 0) critter_mode = 0;
    } else critter_mode = 0;
#ifdef CRITTER
      critter::start(critter_mode);
#endif
      if (A != NULL) {
        assert(b <= A->nrow);
        perturb(A);
#ifdef DEBUG
        if (w.rank == 0)
          printf("A:\n");
        A->print_matrix();
#endif
        if (!b)
          b = ceil(log2(A->nrow));
      }
#ifdef CRITTER
      critter::stop(critter_mode);
#endif
    delete A;
  }
  MPI_Finalize();
  return 0;
}
