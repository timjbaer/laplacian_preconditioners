#include "test.h"

Matrix<REAL> * correct_dist(Matrix<REAL> * A, int b) { // TODO: refactor with correct_ball
  Matrix<REAL> * B = new Matrix<REAL>(A->nrow, A->ncol, A->symm|(A->is_sparse*SP), *(A->wrld), *(A->sr));
  (*B)["ij"] = (*A)["ij"];
  (*B)["ii"] = 0.0;
  for (int i = 0; i < log2(B->nrow); ++i) {
    (*B)["ij"] += (*B)["ik"] * (*B)["kj"];
  }
  return B;
}


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
  int64_t max_ewht=MAX_REAL;
  char *gfile = NULL;
  int64_t n;
  int scale;
  int ef;
  double sp;
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
  if (getCmdOption(input_str, input_str+in_num, "-sp")){
    sp = atof(getCmdOption(input_str, input_str+in_num, "-sp"));
    if (sp < 0.) sp = 0.2;
  } else sp = 0.;
  if (getCmdOption(input_str, input_str+in_num, "-prep")){
    prep = atoll(getCmdOption(input_str, input_str+in_num, "-prep"));
    if (prep < 0) prep = 0;
  } else prep = 0;
  if (getCmdOption(input_str, input_str+in_num, "-seed")){
    seed = atoll(getCmdOption(input_str, input_str+in_num, "-seed"));
    if (seed < 1) seed = 1;
  } else seed = 1;
  srand(seed);

  if (gfile != NULL){ // TODO: have not checked for memory leaks
    int64_t n_nnz = 0;
    if (w.rank == 0)
      printf("Reading real graph n = %lld\n", n);
    A = read_matrix(w, n, gfile, prep, &n_nnz);
    if (w.rank == 0)
      printf("A is on a %dd distribution\n", A->topo->order);
  }
  else if (k != -1) {
    int64_t matSize = pow(3, k);
    if (w.rank == 0)
      printf("Reading kronecker graph n = %lld\n", matSize);
    A = generate_kronecker(&w, k);
    if (w.rank == 0)
      printf("A is on a %dd distribution\n", A->topo->order);
  }
  else if (scale > 0 && ef > 0){
    int64_t n_nnz = 0;
    myseed = SEED;
    if (w.rank == 0)
      printf("R-MAT scale = %d ef = %d seed = %lu\n", scale, ef, myseed);
    A = gen_rmat_matrix(w, scale, ef, myseed, prep, &n_nnz, max_ewht);
    if (w.rank == 0)
      printf("A is on a %dd distribution\n", A->topo->order);
  }
  else if (sp != 0.) {
    int64_t n_nnz = 0;
    if (w.rank == 0)
      printf("uniform matrix n: %lld sparsity: %lf\n", n, sp);
    A = gen_uniform_matrix(w, n, prep, &n_nnz, sp, max_ewht);
    if (w.rank == 0)
      printf("A is on a %dd distribution\n", A->topo->order);
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
  for (int64_t i = 0; i < A_npairs; ++i) {
    A_loc_pairs[i].d += rand() / (REAL) RAND_MAX;
  }
  A->write(A_npairs, A_loc_pairs); 
  delete [] A_loc_pairs;
}

int64_t are_matrices_different(Matrix<REAL> * A, Matrix<REAL> * B)
{
  Scalar<int64_t> s;
  s[""] += Function<REAL,REAL,int64_t>([](REAL a, REAL b){ return (int64_t) fabs(a - b) >= EPSILON; })((*A)["ij"],(*B)["ij"]);
  return s.get_val();
}
