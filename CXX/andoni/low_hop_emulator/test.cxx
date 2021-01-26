#include <ctime>

#ifdef CRITTER
#include "critter.h"
#else
#define TAU_FSTART(ARG)
#define TAU_FSTOP(ARG)
#endif

#include "low_hop_emulator.h"
#include "../../graph.h"

#define EPSILON 0.00001

char* getCmdOption(char ** begin,
                   char ** end,
                   const   std::string & option) {
  char ** itr = std::find(begin, end, option);
  if (itr != end && ++itr != end){
    return *itr;
  }
  return 0;
}

Matrix<REAL> * get_graph(int const in_num, char** input_str, World & w) {
  uint64_t myseed;
  int64_t max_ewht;
  char *gfile = NULL;
  int64_t n;
  int scale;
  int ef;
  int prep;
  int k;
  int seed;

  Matrix<REAL> * A = NULL;
  if (getCmdOption(input_str, input_str+in_num, "-k")) {
    k = atoi(getCmdOption(input_str, input_str+in_num, "-k"));
    if (k < 0) k = 5;
  } else k = -1;
  // K13 : 1594323 (matrix size)
  // K6 : 729; 531441 vertices
  // k5 : 243
  // k7 : 2187
  // k8 : 6561
  // k9 : 19683
  if (getCmdOption(input_str, input_str+in_num, "-f")){
    gfile = getCmdOption(input_str, input_str+in_num, "-f");
  } else gfile = NULL;
  if (getCmdOption(input_str, input_str+in_num, "-n")){
    n = atoll(getCmdOption(input_str, input_str+in_num, "-n"));
    if (n < 0) n = 27;
  } else n = 27;
  if (getCmdOption(input_str, input_str+in_num, "-S")){
    scale = atoi(getCmdOption(input_str, input_str+in_num, "-S"));
    if (scale < 0) scale=10;
  } else scale=0;
  if (getCmdOption(input_str, input_str+in_num, "-E")){
    ef = atoi(getCmdOption(input_str, input_str+in_num, "-E"));
    if (ef < 0) ef=16;
  } else ef=0;
  if (getCmdOption(input_str, input_str+in_num, "-prep")){
    prep = atoll(getCmdOption(input_str, input_str+in_num, "-prep"));
    if (prep < 0) prep = 0;
  } else prep = 0;
  if (getCmdOption(input_str, input_str+in_num, "-seed")){
    seed = atoll(getCmdOption(input_str, input_str+in_num, "-seed"));
    if (seed < 1) seed = 1;
  } else seed = 1;
  srand(seed);

  if (gfile != NULL){
    int n_nnz = 0;
    if (w.rank == 0)
      printf("Reading real graph n = %lld\n", n);
    A = read_matrix(w, n, gfile, prep, &n_nnz);
  }
  else if (k != -1) {
    int64_t matSize = pow(3, k);
    if (w.rank == 0)
      printf("Reading kronecker graph n = %lld\n", matSize);
    A = generate_kronecker(&w, k);
  }
  else if (scale > 0 && ef > 0){
    int n_nnz = 0;
    myseed = SEED;
    if (w.rank == 0)
      printf("R-MAT scale = %d ef = %d seed = %lu\n", scale, ef, myseed);
    A = gen_rmat_matrix(w, scale, ef, myseed, prep, &n_nnz, max_ewht);
  }
  else {
    if (w.rank == 0) {
      printf("No graph specified\n");
    }
  }
  return A;
}

void perturb(Matrix<REAL> * A) {
  int64_t A_npairs;
  Pair<REAL> * A_loc_pairs;
  A->get_local_pairs(&A_npairs, &A_loc_pairs, true);
  for (int i = 0; i < A_npairs; ++i) {
    if (A_loc_pairs[i].d != 0) { // TODO: should be unnecessary if get_local_pairs only returns nonzeros
      A_loc_pairs[i].d += rand() / (REAL) RAND_MAX;
    }
  }
  A->write(A_npairs, A_loc_pairs); 
}

Matrix<bpair> * real_to_bpair(Matrix<REAL> * A) {
  int n = A->nrow;
  const static Monoid<bpair> MIN_BPAIR = get_bpair_monoid();
  Matrix<bpair> * B = new Matrix<bpair>(n, n, *(A->wrld), MIN_BPAIR);
  int64_t npairs;
  Pair<REAL> * A_loc_pairs;
  A->get_local_pairs(&npairs, &A_loc_pairs, true);
  Pair<bpair> B_loc_pairs[npairs];
  for (int i = 0; i < npairs; ++i) {
    B_loc_pairs[i].k = A_loc_pairs[i].k;
    B_loc_pairs[i].d.vertex = A_loc_pairs[i].k % n;
    B_loc_pairs[i].d.dist = A_loc_pairs[i].d;
  }
  B->write(npairs, B_loc_pairs);
  delete [] A_loc_pairs;
  return B;
}

// void test_ball_red() { // run on one process
//   printf("test ball reduction\n");
//   int n = 1;
//   int b = 5;
//   ball_t * x = (ball_t *) malloc(sizeof(ball_t) + n * b * sizeof(Pair<REAL>));
//   ball_t * y = (ball_t *) malloc(sizeof(ball_t) + n * b * sizeof(Pair<REAL>));
//   x->n = n;
//   y->n = n;
//   x->b = b;
//   y->b = b;
//   for (int64_t i = 0; i < b; ++i) {
//     x->closest_neighbors[i].k = 2 * i;
//     y->closest_neighbors[i].k = 2 * i + 1;
//     x->closest_neighbors[i].d = 2 * i;
//     y->closest_neighbors[i].d = 2 * i + 1;
//   }
//   ball_red(x, y, 1);
//   int pass = 1;
//   for (int i = 0; i < b; ++i) {
//     printf("%f ", y->closest_neighbors[i].d);
//     if (y->closest_neighbors[i].d != i)
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

// ball_t * correct_ball(Matrix<REAL> * A, int b) {
//   Matrix<REAL> * B = new Matrix<REAL>(*A);
//   for (int i = 0; i < log2(B->nrow); ++i) {
//     (*B)["ij"] += (*B)["ik"] * (*B)["kj"];
//   }
//   ball_t * correct = filter(B, b);
//   delete B;
//   return correct;
// }

int64_t are_matrices_different(Matrix<REAL> * A, Matrix<REAL> * B) // TODO: may need to consider sparsity like mst are_vectors_different()
{
  Scalar<int64_t> s;
  s[""] += Function<REAL,REAL,int64_t>([](REAL a, REAL b){ return (int64_t) fabs(a - b) >= EPSILON; })((*A)["ij"],(*B)["ij"]);
  return s.get_val();
}

Matrix<REAL> * correct_ball(Matrix<REAL> * A, int b) { // assumes correct of filter()
  int n = A->nrow;
  int symm = A->symm;
  World wrld = *(A->wrld);
  Matrix<REAL> * B = new Matrix<REAL>(*A);
  for (int i = 0; i < log2(B->nrow); ++i) {
    (*B)["ij"] += (*B)["ik"] * (*B)["kj"];
  }
  ball_t * ball = filter(B, b);
  delete B;
  B = new Matrix<REAL>(n, n, symm, wrld, MIN_PLUS_SR);
  B->write(n*b, ball->closest_neighbors);
  return B;
}

int64_t check_ball(Matrix<REAL> * A, Matrix<REAL> * B, int b) {
  Matrix<REAL> * correct = correct_ball(A, b);
  // printf("A\n");
  // A->print_matrix();
  // printf("B\n");
  // B->wrld = A->wrld;
  // B->print_matrix();
  // printf("correct\n");
  // correct->print_matrix();
  B->wrld = A->wrld; // FIXME: why does B's world become NULL?
  correct->wrld = A->wrld; // FIXME: why does correct's world become NULL?
  int64_t s = are_matrices_different(correct, B);
  if (A->wrld->rank == 0)
    printf("correct:\n");
  correct->print_matrix();
  delete correct;
  return s;
}

// void compare_ball(Matrix<REAL> * A, Matrix<REAL> * B, int b) { // TODO: assumes B has at most b nonzeros per row
//   int n = A->nrow;
//   printf("print B (important)\n");
//   B->print_matrix();
//   exit(1);
//   ball_t * ball = filter(B, b);
//   ball_t * correct = correct_ball(A, b);
//   assert(ball->n == correct->n);
//   assert(ball->b == correct->b);
//   int diff = 0;
//   if (A->wrld->rank == 0) {
// #ifdef DEBUG
//     printf("ball:\n");
//     for (int i = 0; i < n; ++i) {
//       for (int j = 0; j < b; ++j) {
//         printf("(%d %f) ", ball->closest_neighbors[i*b + j].k, ball->closest_neighbors[i*b + j].d);
//       }
//       printf("\n");
//     }
// 
//     printf("correct ball:\n");
//     for (int i = 0; i < n; ++i) {
//       for (int j = 0; j < b; ++j) {
//         printf("(%d %f) ", correct->closest_neighbors[i*b + j].k, correct->closest_neighbors[i*b + j].d);
//       }
//       printf("\n");
//     }
// #endif
// 
//     for (int i = 0; i < n; ++i) {
//       for (int j = 0; j < b; ++j) {
//         if (ball->closest_neighbors[i*b + j].k != correct->closest_neighbors[i*b + j].k
//          || ball->closest_neighbors[i*b + j].d != correct->closest_neighbors[i*b + j].d) {
//           ++diff;
//         }
//       }
//     }
//     if (!diff) {
//       printf("passed test ball reduction\n");
//     } else {
//       printf("failed test ball reduction, differ by %d\n", diff);
//     }
//   }
//   free(correct);
//   free(ball);
// }

int main(int argc, char** argv)
{
  int const in_num = argc;
  char** input_str = argv;
  int critter_mode=0;
  int b;
  int bvec=0;

  int rank;
  int np;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  {
    World w(argc, argv);
    Matrix<REAL> * B = get_graph(argc, argv, w);
    Matrix<REAL> * A = new Matrix<REAL>(B->nrow, B->ncol, B->symm, w, MIN_PLUS_SR);
    (*A)["ij"] = (*B)["ij"]; // change to (min, +) semiring
    (*A)["ii"] = 0;
    delete B;
    if (getCmdOption(input_str, input_str+in_num, "-b")){
      b = atoi(getCmdOption(input_str, input_str+in_num, "-b"));
      if (b < 0) b = 0;
    }
    if (getCmdOption(input_str, input_str+in_num, "-bvec")){
      bvec = atoi(getCmdOption(input_str, input_str+in_num, "-bvec"));
      if (bvec < 0) bvec = 0;
    } else bvec = 0;
    if (getCmdOption(input_str, input_str+in_num, "-critter_mode")){
      critter_mode = atoi(getCmdOption(input_str, input_str+in_num, "-critter_mode"));
      if (critter_mode < 0) critter_mode = 0;
    } else critter_mode = 0;
    init_mpi(A->nrow, b);
#ifdef CRITTER
      critter::start(critter_mode);
#endif
      if (A != NULL) {
        perturb(A);
#ifdef DEBUG
        if (w.rank == 0)
          printf("A:\n");
        A->print_matrix();
#endif
        if (!b)
          b = ceil(log2(A->nrow));
        if (!bvec) {
          TAU_FSTART(ball via matmat);
          double stime = MPI_Wtime();
          Matrix<REAL> * ball = ball_matmat(A, b);
          double etime = MPI_Wtime();
          TAU_FSTOP(ball via matmat);
#ifdef DEBUG
          if (w.rank == 0)
            printf("ball (via matmat):\n");
          ball->print_matrix();
          // compare_ball(A, ball, b); // FIXME: causes MPI null ptr excep
          int64_t diff = check_ball(A, ball, b);
          if (w.rank == 0)
            printf("ball (via matmat) diff: %" PRId64 "\n", diff);
#endif
          delete ball;
          if (w.rank == 0)
            printf("ball (via matmat) done in %1.2lf\n", etime - stime);
        } else {
          TAU_FSTART(ball via matvec);
          double stime = MPI_Wtime();
          Matrix<bpair> * B = real_to_bpair(A);
          Vector<bvector<BALL_SIZE>> * ball = ball_bvector<BALL_SIZE>(B);
          double etime = MPI_Wtime();
          TAU_FSTOP(ball via matvec);
#ifdef DEBUG
          if (w.rank == 0)
            printf("ball (via matvec):\n");
          ball->print();
#endif
          delete ball;
          delete B;
          if (w.rank == 0)
            printf("ball (via matvec) done in %1.2lf\n", etime - stime);
        }
      }
#ifdef CRITTER
      critter::stop(critter_mode);
#endif
    destroy_mpi();
    delete A;
  }
  MPI_Finalize();
  return 0;
}
