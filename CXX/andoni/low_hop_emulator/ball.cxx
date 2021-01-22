#include "ball.h"

/***** filter b closest neighbors *****/
MPI_Datatype MPI_PAIR;
MPI_Datatype MPI_BALL;
MPI_Op oball;

void create_mpi_pair() {
  int lens[2]= { 1, 1 };
  const MPI_Aint disps[2] = { 0, sizeof(int64_t) };
  MPI_Datatype types[2] = { MPI_INT64_T, MPI_FLOAT };
  MPI_Type_create_struct(2, lens, disps, types, &MPI_PAIR);
  MPI_Type_commit(&MPI_PAIR);
}

void create_mpi_ball(int n, int b) {
  create_mpi_pair();
  int lens[3]= { 1, 1, n * b };
  const MPI_Aint disps[3] = { 0, sizeof(int), 2 * sizeof(int) };
  MPI_Datatype types[3] = { MPI_INT, MPI_INT, MPI_PAIR };
  MPI_Type_create_struct(3, lens, disps, types, &MPI_BALL);
  MPI_Type_commit(&MPI_BALL);
}

void ball_red(ball_t const * x,
              ball_t * y,
              int nitems);

bool is_init_mpi = false;
void init_mpi(int n, int b) {
  create_mpi_pair();
  create_mpi_ball(n, b);
  MPI_Op_create(
      [](void * x, void * y, int * n, MPI_Datatype*){
        ball_red((ball_t*) x, (ball_t*) y, *n);
      },
      1,
      &oball);
}

void destroy_mpi() {
  MPI_Op_free(&oball);
  MPI_Type_free(&MPI_BALL);
  MPI_Type_free(&MPI_PAIR);
}

ball_t * filter(Matrix<REAL> * A, int b) {
  // assert(A->symm != SY);
  // assert(A->edge_map[0].type == CTF_int::PHYSICAL_MAP);
  int n = A->nrow;
  if (!is_init_mpi) {
    init_mpi(n, b);
    is_init_mpi = true;
  }
  int64_t A_npairs;
  Pair<REAL> * A_loc_pairs;
  A->get_local_pairs(&A_npairs, &A_loc_pairs, true);
  ball_t * ball = (ball_t *) malloc(sizeof(ball_t) + n * b * sizeof(Pair<REAL>));
  ball->n = n;
  ball->b = b;
  for (int64_t i = 0; i < n; ++i) {
    for (int64_t j = 0; j < b; ++j) {
      ball->closest_neighbors[i*b + j].k = -1; // FIXME: issue if a vertex has less than b neighbors
      ball->closest_neighbors[i*b + j].d = MAX_REAL;
    }
  }
  for (int64_t i = 0; i < A_npairs; ++i) {
    int vertex = A_loc_pairs[i].k / n;
    REAL dist = A_loc_pairs[i].d;
    if (dist < ball->closest_neighbors[vertex*b + b-1].d) { // TODO: not efficient
      ball->closest_neighbors[vertex*b + b-1] = A_loc_pairs[i];
      std::sort(ball->closest_neighbors + vertex*b, ball->closest_neighbors + vertex*b + b,
                [](Pair<REAL> const & first, Pair<REAL> const & second) -> bool
                  { return first.d < second.d; }
                );
    }
  }

  // CTF_int::CommData row_commdata = A->topo->dim_comm[A->edge_map[0].cdt];
  // row_commdata.activate(MPI_COMM_WORLD);
  // MPI_Comm row_comm = row_commdata.cm;
  // MPI_Allreduce(MPI_IN_PLACE, ball, 1, MPI_BALL, oball, row_comm);
  // row_commdata.allred(MPI_IN_PLACE, ball, 1, MPI_BALL, oball);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  char * idx = NULL;
  Idx_Partition prl;
  Idx_Partition blk;
  A->get_distribution(&idx, prl, blk);
  int color = 1; // TODO: communicate across rows

  MPI_Comm row_comm;
  MPI_Comm_split(MPI_COMM_WORLD, color, rank, &row_comm);
  MPI_Allreduce(MPI_IN_PLACE, ball, 1, MPI_BALL, oball, row_comm);

  delete [] A_loc_pairs;

  return ball;
}

ball_t * filter(Matrix<bpair> * A, int b) {
  // assert(A->symm != SY);
  // assert(A->edge_map[0].type == CTF_int::PHYSICAL_MAP);
  int n = A->nrow;
  if (!is_init_mpi) {
    init_mpi(n, b);
    is_init_mpi = true;
  }
  int64_t A_npairs;
  Pair<bpair> * A_loc_pairs;
  A->get_local_pairs(&A_npairs, &A_loc_pairs, true);
  ball_t * ball = (ball_t *) malloc(sizeof(ball_t) + n * b * sizeof(Pair<REAL>));
  ball->n = n;
  ball->b = b;
  for (int64_t i = 0; i < n; ++i) {
    for (int64_t j = 0; j < b; ++j) {
      ball->closest_neighbors[i*b + j].k = -1; // FIXME: issue if a vertex has less than b neighbors
      ball->closest_neighbors[i*b + j].d = MAX_REAL;
    }
  }
  for (int64_t i = 0; i < A_npairs; ++i) {
    int vertex = A_loc_pairs[i].d.vertex;
    REAL dist = A_loc_pairs[i].d.dist;
    if (dist < ball->closest_neighbors[vertex*b + b-1].d) { // TODO: not efficient
      ball->closest_neighbors[vertex*b + b-1].k = A_loc_pairs[i].k;
      ball->closest_neighbors[vertex*b + b-1].d = A_loc_pairs[i].d.dist;
      std::sort(ball->closest_neighbors + vertex*b, ball->closest_neighbors + vertex*b + b,
                [](Pair<REAL> const & first, Pair<REAL> const & second) -> bool
                  { return first.d < second.d; }
                );
    }
  }

  // CTF_int::CommData row_commdata = A->topo->dim_comm[A->edge_map[0].cdt];
  // row_commdata.activate(MPI_COMM_WORLD);
  // MPI_Comm row_comm = row_commdata.cm;
  // MPI_Allreduce(MPI_IN_PLACE, ball, 1, MPI_BALL, oball, row_comm);
  // row_commdata.allred(MPI_IN_PLACE, ball, 1, MPI_BALL, oball);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  char * idx = NULL;
  Idx_Partition prl;
  Idx_Partition blk;
  A->get_distribution(&idx, prl, blk);
  int color = 1; // TODO: communicate across rows

  MPI_Comm row_comm;
  MPI_Comm_split(MPI_COMM_WORLD, color, rank, &row_comm);
  MPI_Allreduce(MPI_IN_PLACE, ball, 1, MPI_BALL, oball, row_comm);

  delete [] A_loc_pairs;

  return ball;
}


/***** matmat approach *****/
int search(REAL target, const Pair<REAL> * y, int l, int r) { // TODO: implement binary search
  int n = r - l;
  for (int i = 0; i < n; ++i) {
    if (target < y[i].d) {
      return i;
    }
  }
  return n;
}

void merge(Pair<REAL> const * x,
                   Pair<REAL> * y,
                   int b) { // FIXME: requires elements of x \cup y to be distinct (currently fixing with perturb())
  int idx[2*b];
  Pair<REAL> * y_prev = (Pair<REAL> *) malloc(b * sizeof(Pair<REAL>));
#ifdef _OPENMP
  #pragma omp parallel for
#endif
  for (int i = 0; i < b; ++i) {
    idx[i] = i + search(x[i].d, y, 0, b);
    idx[i+b] = i + search(y[i].d, x, 0, b);
    y_prev[i] = y[i];
  }
#ifdef _OPENMP
  #pragma omp parallel for
#endif
  for (int i = 0; i < 2*b; ++i) {
    if (idx[i] < b) {
      if (i < b) {
        y[idx[i]] = x[i];
      } else {
        y[idx[i]] = y_prev[i-b];
      }
    }
  }
}

void ball_red(ball_t const * x,
              ball_t * y,
              int nitems) { // TODO: use sparsity
  assert(nitems == 1);
  assert(x->n == y->n);
  assert(x->b == y->b);
  int n = x->n;
  int b = x->b;
#ifdef _OPENMP
  #pragma omp parallel for
#endif
  for (int vertex = 0; vertex < n; ++vertex) {
    merge(x->closest_neighbors + vertex*b, y->closest_neighbors + vertex*b, b);
  }
}

Matrix<REAL> * ball_matmat(Matrix<REAL> * A, int64_t b) { // A should be on (min, +) semiring
  int n = A->nrow;
  Matrix<REAL> * B = new Matrix<REAL>(*A);
  // for (int i = 0; i < log2(B->nrow); ++i) {
  //   (*B)["ij"] += (*B)["ik"] * (*B)["kj"];
  // }
  Matrix<REAL> * C = new Matrix<REAL>(n, n, B->symm, *(B->wrld), MIN_PLUS_SR);
  ball_t * ball = filter(B, b);
  delete B;
  C->write(n*b, ball->closest_neighbors);

  destroy_mpi();
  return C;
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
