#include "test_ball.h"

//===================================================================
// Test Ball
//===================================================================
// Input: graph
// Output: number of incorrect closest neighbors
//
// This tests the b-closest neighbor functions that use
// 1. matrix format only
//   a. path doubling (ball_matmat)
// 2. matrix-vector format
//   b. single ball update (ball_bvector)
//   c. multilinear (ball_multilinear)
// and compares the b closest neighbors derived from these against
// the true b closest neighbors for all vertices
//
// ------------------------------------------------------------------
//
// To run:
//
// mpirun -n 4 ./run_test_ball <graph flags>
//
// optional flags are:
// -critter_mode                      --> disabled
//
// -b [Z]                             number of closest neighbors
//
// -bvec 1 | -multi 1                 run bvector or multilinear
//
// -d [1,2]                           1D or 2D data partition layout
//
// -conv                              detects early convergence
//
// -square                            early update of A that changes
//                                     B(i+1,u) = min+_k(B(i,u), A_kv)
//                                     into
//                                     B(i+1,u)=min+_k(B(i,u), B(i,k))
//                                     or min+_k(B(i,u),B(i+1,k))
//                                     depending on order
//
//===================================================================


int main(int argc, char** argv)
{
  int const in_num = argc;
  char** input_str = argv;

  // flags
  int critter_mode=0; // disabled
  int b=BALL_SIZE; // *
  int bvec=0;
  int multi=0;
  int d=2;
  int conv=0;
  int square=0;

  // MPI
  int rank;
  int np;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  {
    // CTF
    World w(argc, argv);

    // parse flags --------------------------------------------------

    if (getCmdOption(input_str, input_str+in_num, "-d")){
      d = atoi(getCmdOption(input_str, input_str+in_num, "-d"));
      if (d < 1 || d > 2) d = 2;
    }

    if (getCmdOption(input_str, input_str+in_num, "-b")){
      b = atoi(getCmdOption(input_str, input_str+in_num, "-b"));
      if (b < 0) b = 0;
    }

    if (getCmdOption(input_str, input_str+in_num, "-bvec")){
      bvec = atoi(getCmdOption(input_str, input_str+in_num, "-bvec"));
      if (bvec < 0) bvec = 0;
    } else bvec = 0;

    if (getCmdOption(input_str, input_str+in_num, "-multi")){
      multi = atoi(getCmdOption(input_str, input_str+in_num, "-multi"));
      if (multi < 0) multi = 0;
    } else multi = 0;

    assert(!multi || !bvec);
    if (rank == 0) {
      if (bvec > 0) { printf("Using bvector...\n");
      } else if (multi > 0) { printf("Using multilinear...\n");
      } else { printf("Using matmat...\n"); }
    }

    if (getCmdOption(input_str, input_str+in_num, "-conv")){
      conv = atoi(getCmdOption(input_str, input_str+in_num, "-conv"));
      if (conv < 0) conv = 0;
    } else conv = 0;

    if (getCmdOption(input_str, input_str+in_num, "-square")){
      square = atoi(getCmdOption(input_str, input_str+in_num, "-square"));
      if (square < 0) square = 0;
    } else square = 0;

    if (getCmdOption(input_str, input_str+in_num, "-critter_mode")){
      critter_mode = atoi(getCmdOption(input_str, input_str+in_num, "-critter_mode"));
      if (critter_mode < 0) critter_mode = 0;
    } else critter_mode = 0;

#ifdef CRITTER
      critter::start(critter_mode);
#endif

    // create ball computation matrix from input graph  -------------

    Matrix<REAL> * B = get_graph(argc, argv, w); // temp input graph
    Matrix<REAL> * A; // ball computation matrix

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

    // run test -----------------------------------------------------
    // spacing *

      if (A != NULL) {
        if (b > A->nrow) b = A->nrow; // *

        perturb(A); // for each existing edge add rand len btwn [0,1]

#ifdef DEBUG
        if (w.rank == 0) printf("A:\n");
        A->print_matrix();
#endif

        if (!b) b = ceil(log2(A->nrow)); // *

        if (w.rank == 0) {
          run_test_ball(d, A, bvec, multi, conv, square, b, true);
        } else {
          run_test_ball(d, A, bvec, multi, conv, square, b, false);
        }
      }
#ifdef CRITTER
      critter::stop(critter_mode);
#endif
    delete A;
  }

  MPI_Finalize();
  return 0;
}
