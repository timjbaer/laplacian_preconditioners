#include "test.h"
#include "low_hop_emulator.h"

Matrix<REAL> * correct_dist(Matrix<REAL> * A, int b) { // TODO: refactor with correct_ball
  Matrix<REAL> * B = new Matrix<REAL>(A->nrow, A->ncol, A->symm|(A->is_sparse*SP), *(A->wrld), *(A->sr));
  (*B)["ij"] = (*A)["ij"];
  for (int i = 0; i < log2(B->nrow); ++i) {
    (*B)["ij"] += (*B)["ik"] * (*B)["kj"];
  }
  return B;
}

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
    //   A = new Matrix<REAL>(B->nrow, B->ncol, "ij", part["i"], blk, B->symm|(B->is_sparse*SP), w, MIN_PLUS_SR);
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

        Subemulator subem(A, b);
#ifdef DEBUG
        Matrix<REAL> * H = subem.H;
        Vector<bpair> * q = subem.q;
        if (w.rank == 0)
          printf("subemulator H:\n");
        H->print_matrix();
        int64_t A_nnz = A->nnz_tot;
        int64_t H_nnz = H->nnz_tot;
        Matrix<REAL> * A_dist = correct_dist(A, b);
        Matrix<REAL> * H_dist = correct_dist(H, b);
        Matrix<REAL> * D = new Matrix<REAL>(A->nrow, A->ncol, A->symm|(A->is_sparse*SP), w, MAX_MONOID); // MAX_MONOID is a hack to avoid Transform accumulating
        Bivar_Function<REAL,REAL,REAL> distortion([](REAL x, REAL y){ return x / y; });
        distortion.intersect_only = true;
        (*D)["ij"] = distortion((*H_dist)["ij"], (*A_dist)["ij"]);
        double max_distort = D->norm_infty();
        Scalar<REAL> tot_distort(w);
        tot_distort[""] = Function<REAL,REAL>([](REAL x){ return x; })((*D)["ij"]);
        double avg_distort = tot_distort.get_val() / D->nnz_tot;
        D->set_zero();
        Transform<REAL>([](REAL & d){ d *= 22; })((*D)["ij"]);
        Transform<bpair,REAL>([](bpair pr, REAL & d){ d += pr.dist; })((*q)["i"], (*D)["ij"]);
        Transform<bpair,REAL>([](bpair pr, REAL & d){ d += pr.dist; })((*q)["j"], (*D)["ij"]);
        Scalar<int64_t> count(w);
        count[""] = Function<REAL,REAL,int64_t>([](REAL x, REAL y){ return (int64_t)(x > y); })((*D)["ij"], (*A_dist)["ij"]);
        int64_t cnt = count.get_val();
        delete H_dist;
        delete A_dist;
        delete D;
        if (w.rank == 0) {
          printf("subemulator has %" PRId64 " nonzeros\n", H_nnz);
          int check = (int) (H_nnz <= 0.75 * A_nnz);
          if (check)
            printf("passed: subemulator has less than 0.75n edges\n");
          else 
            printf("failed: subemulator has more than 0.75n edges\n");

          printf("subemulator max distance distortion between sampled vertices is %f\n", max_distort);
          printf("subemulator avg distance distortion between sampled vertices is %f\n", avg_distort);
          check = (int) (max_distort >= 1 && max_distort <= 8);
          if (check)
            printf("passed: subemulator preserves distances among sampled vertices\n");
          else 
            printf("failed: subemulator does not preserve distances among sampled vertices\n");

          if (cnt == 0)
            printf("passed: subemulator preserves distances among vertices\n");
          else 
            printf("failed: subemulator does not preserve distances among vertices\n");
        }
#endif
      }
#ifdef CRITTER
      critter::stop(critter_mode);
#endif
    delete A;
  }
  MPI_Finalize();
  return 0;
}
