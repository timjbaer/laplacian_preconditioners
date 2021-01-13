#include <ctime>

#ifdef CRITTER
#include "critter.h"
#else
#define TAU_FSTART(ARG)
#define TAU_FSTOP(ARG)
#endif

#include "low_hop_emulator.h"
#include "../../graph.h"


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
    if (A_loc_pairs[i].d != 0) {
      A_loc_pairs[i].d += rand() / (REAL) RAND_MAX;
    }
  }
  A->write(A_npairs, A_loc_pairs); 
}

void test_ball_red() { // run on one process
  printf("testing test ball reduction\n");
  int n = 1;
  int b = 5;
  ball_t * x = (ball_t *) malloc(sizeof(ball_t) + n * b * sizeof(Pair<REAL>));
  ball_t * y = (ball_t *) malloc(sizeof(ball_t) + n * b * sizeof(Pair<REAL>));
  x->n = n;
  y->n = n;
  x->b = b;
  y->b = b;
  for (int64_t i = 0; i < b; ++i) {
    x->closest_neighbors[i].k = 2 * i;
    y->closest_neighbors[i].k = 2 * i + 1;
    x->closest_neighbors[i].d = 2 * i;
    y->closest_neighbors[i].d = 2 * i + 1;
  }
  ball_red(x, y, 1);
  int pass = 1;
  for (int i = 0; i < b; ++i) {
    printf("%f ", y->closest_neighbors[i].d);
    if (y->closest_neighbors[i].d != i)
      pass = 0;
  }
  printf("\n");
  if (pass) {
    printf("passed test ball reduction\n");
  } else {
    printf("failed test ball reduction\n");
  }
  free(x);
  free(y);
}

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
#ifdef DEBUG
          if (w.rank == 0)
            printf("ball:\n");
          ball->print_matrix();
#endif
          delete ball;
          double etime = MPI_Wtime();
          TAU_FSTOP(ball via matmat);
          if (w.rank == 0)
            printf("ball via matmat done in %1.2lf\n", etime - stime);
        } else {
          TAU_FSTART(ball via matvec);
          double stime = MPI_Wtime();
          Vector<bvector<BALL_SIZE>> * ball = ball_bvector<BALL_SIZE>(A);
#ifdef DEBUG
          if (w.rank == 0)
            printf("ball:\n");
          ball->print();
#endif
          delete ball;
          double etime = MPI_Wtime();
          TAU_FSTOP(ball via matvec);
          if (w.rank == 0)
            printf("ball via matvec done in %1.2lf\n", etime - stime);
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
