#ifndef TEST_BALL_H
#define TEST_BALL_H

#include "test.h"
#include "../ball.h"

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

// creates the b-closest neighbor matrix using a "verified" method
Matrix<REAL> * correct_ball(Matrix<REAL> * A, int b);

// checks if B matches the correct b-closest neighbor matrix
int64_t check_ball(Matrix<REAL> * A, Matrix<REAL> * B, int b);
int64_t check_ball(Matrix<REAL> * A, Vector<bvector<BALL_SIZE>> * B, int b);

// runs the ball computation process using bvectors, matmat, or multilinear
void run_test_ball(int d, Matrix<REAL> * A, int bvec, int multi,
        int conv, int square, int b=BALL_SIZE, bool canPrint=false);

#endif //TEST_BALL_H
