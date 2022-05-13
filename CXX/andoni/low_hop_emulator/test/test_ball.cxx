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
//===================================================================


/* For a graph G(V,E), creates a 1D layout symmetric matrix
 * that for each row r contains the b closest neighbors of vertex
 * r
 *
 * we compute b closest neighbors using log2(|V|) iterations
 * of path doubling and then filtering each row to have the closest b
 */
Matrix<REAL> * correct_ball(Matrix<REAL> * A, int b) {
  int np = A->wrld->np;
  int plens[1] = { np };
  Partition part(1, plens);
  Idx_Partition blk;
  Matrix<REAL> * B = new Matrix<REAL>(A->nrow, A->ncol, "ij", part["i"], blk, A->symm|(A->is_sparse*SP), *(A->wrld), *(A->sr));
  (*B)["ij"] = (*A)["ij"]; // change to correct distribution
  (*B)["ii"] = 0.0;
  for (int i = 0; i < log2(B->nrow); ++i) {
    (*B)["ij"] += (*B)["ik"] * (*B)["kj"];
  }
  filter(B, b); // assumes correct filter()
  return B;
}

//-------------------------------------------------------------------

/* if B is not matrix, then converts vector of balls for each
 * vertex B to matrix B and then checks each element between
 * A and B, returns 1 if different else 0
 */
int64_t check_ball(Matrix<REAL> * A, Matrix<REAL> * B, int b) {
  Matrix<REAL> * correct = correct_ball(A, b);
  // if (A->wrld->rank == 0)
  //   printf("correct:\n");
  // correct->print_matrix();
  int64_t s = are_matrices_different(correct, B);
  delete correct;
  return s;
}
 
int64_t check_ball(Matrix<REAL> * A, Vector<bvector<BALL_SIZE>> * B, int b) {
  int n = A->nrow;
  Matrix<REAL> * C = new Matrix<REAL>(n, n, A->symm|(A->is_sparse*SP), *(A->wrld), *(A->sr));
  bvec_to_mat(C, B);
  int64_t s = check_ball(A, C, b);
  delete C;
  return s;
}

//-------------------------------------------------------------------

void run_test_ball(int d, Matrix<REAL> * A, int bvec, int multi,
        int conv, int square, int b, bool canPrint) {

  // compute b-closest neighbors of all vertices
  // by path doubling using matrix format
  if (!bvec && !multi) {
    TAU_FSTART(ball via matmat);
    double stime = MPI_Wtime();
    Matrix<REAL> * ball = ball_matmat(A, b);
    double etime = MPI_Wtime();
    TAU_FSTOP(ball via matmat);

#ifdef DEBUG
    if (canPrint) printf("ball (via matmat):\n");
    ball->print_matrix(); // *
#endif

#if defined TEST || defined DEBUG
    // compare results from b-closest neighbor function
    // to true b-closest neighbors
    int64_t diff = check_ball(A, ball, b);
    if (canPrint)
      printf("ball (via matmat) diff: %" PRId64 "\n", diff);
#endif

    delete ball;

    if (canPrint)
      printf("ball (via matmat) done in %1.2lf\n", etime - stime);


  } else {
  // compute b-closest neighbors of all using vector
  // methods such as single ball update or using
  // multilinear format
    TAU_FSTART(ball via matvec);
    double stime = MPI_Wtime();
    Vector<bvector<BALL_SIZE>> * ball = nullptr;
    if (bvec)
      ball = ball_bvector<BALL_SIZE>(A,conv,square);
    else if (multi)
      ball = ball_multilinear<BALL_SIZE>(A,conv,square);
    double etime = MPI_Wtime();
    TAU_FSTOP(ball via matvec);

#ifdef DEBUG
    if (canPrint) printf("ball (via matvec):\n");
    ball->print(); // *
#endif

#if defined TEST || defined DEBUG
    // compare results from b-closest neighbor function
    // to true b-closest neighbors
    int64_t diff = check_ball(A, ball, b);
    if (canPrint)
      printf("ball (via matvec) diff: %" PRId64 "\n", diff);
#endif

    delete ball;

    if (canPrint)
      printf("ball (via matvec) done in %1.2lf\n", etime - stime);
  }
}

//-------------------------------------------------------------------

// void test_bvector_red() { // run on one process
//   printf("test bvector reduction\n");
//   bvector<10> * x = (bvector<10> *) malloc(sizeof(bvector<10>));
//   bvector<10> * y = (bvector<10> *) malloc(sizeof(bvector<10>));
//   for (int64_t i = 0; i < 5; ++i) {
//     x->closest_neighbors[i].vertex = 2 * i;
//     y->closest_neighbors[i].vertex = 2 * i + 1;
//     x->closest_neighbors[i].dist = 2 * i;
//     y->closest_neighbors[i].dist = 2 * i + 1;
//   }
//   for (int i = 0; i < 5; ++i) { // duplicate vertices
//     y->closest_neighbors[5+i].vertex = 2 * i;
//     x->closest_neighbors[5+i].vertex = 2 * i + 1;
//     y->closest_neighbors[5+i].dist = 2 * i + 1;
//     x->closest_neighbors[5+i].dist = 2 * i + 1 + 1;
//   }
//   bvector_red(x, y, 1);
//   int pass = 1;
//   for (int i = 0; i < 10; ++i) {
//     printf("%f ", y->closest_neighbors[i].dist);
//     if (y->closest_neighbors[i].dist != i)
//       pass = 0;
//   }
//   printf("\n");
//   if (pass) {
//     printf("passed test ball reduction\n");
//   } else {
//     printf("failed test ball reduction\n");
//   }
//   free(x);
//   free(y);
// }
