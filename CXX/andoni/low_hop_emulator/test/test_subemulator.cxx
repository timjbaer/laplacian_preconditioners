#include "test.h"
#include "../subemulator.h"

void test_connects(World & w) { // run on 4 processes
  int b = 4;
  int64_t n_nnz = 0;
  Matrix<REAL> * B = gen_rmat_matrix(w, 3, 8, 23, 0, &n_nnz, MAX_REAL);
  Matrix<REAL> * A = new Matrix<REAL>(B->nrow, B->ncol, B->symm|(B->is_sparse*SP), w, MIN_PLUS_SR);
  (*A)["ij"] = (*B)["ij"]; // change to (min, +) semiring and correct distribution
  delete B;
  if (w.rank == 0)
    printf("A:\n");
  A->print_matrix();
  assert(A->is_sparse); // not strictly necessary, but much more efficient

  Subemulator subem(A, b);
  Matrix<REAL> * H = subem.H;
  if (w.rank == 0)
    printf("H:\n");
  H->print_matrix();
  
  Monoid<bpair> bpair_monoid = get_bpair_monoid();
  Vector<bpair> q(8, w, bpair_monoid);
  int64_t q_nprs = 8;
  Pair<bpair> q_prs[q_nprs];
  q_prs[0].k = 0;
  q_prs[0].d = bpair(2, 33.000000);
  q_prs[1].k = 1;
  q_prs[1].d = bpair(2, 72.000000);
  q_prs[2].k = 2;
  q_prs[2].d = bpair(2, MAX_REAL);
  q_prs[3].k = 3;
  q_prs[3].d = bpair(2, 67.000000);
  q_prs[4].k = 4;
  q_prs[4].d = bpair(6, 183.000000);
  q_prs[5].k = 5;
  q_prs[5].d = bpair(6, 77.000000);
  q_prs[6].k = 6;
  q_prs[6].d = bpair(6, MAX_REAL);
  q_prs[7].k = 7;
  q_prs[7].d = bpair(1, 10.000000);
  q.write(q_nprs, q_prs);

  Scalar<int> q_diff;
  q_diff[""] = Function<bpair,bpair,int>([](bpair first, bpair second){ return (int)(first.vertex != second.vertex || fabs(first.dist - second.dist) >= 1); })(q["i"], (*subem.q)["i"]);
  if (q_diff.get_val() > 0) {
    if (w.rank == 0)
      printf("q is wrong, exiting...\n");
    exit(1);
  }

  Matrix<REAL> C(8, 8, H->symm|(H->is_sparse*SP), w, MIN_PLUS_SR);
  int64_t nprs = 7;
  Pair<REAL> prs[nprs];
  prs[0].k = 10;
  prs[0].d = 92;
  prs[1].k = 14;
  prs[1].d = 405;
  prs[2].k = 17;
  prs[2].d = 92;
  prs[3].k = 18;
  prs[3].d = 189;
  prs[4].k = 22;
  prs[4].d = 518;
  prs[5].k = 49;
  prs[5].d = 405;
  prs[6].k = 50;
  prs[6].d = 518;
  prs[7].k = 54;
  prs[7].d = 366;
  C.write(nprs, prs);

  if (w.rank == 0)
    printf("C:\n");
  C.print_matrix();

  Scalar<int64_t> diff;
  diff[""] = Function<REAL,REAL,int64_t>([](REAL h, REAL c){ return (int64_t)(fabs(h - c) >= 1); })((*H)["ij"], C["ij"]);
  int64_t val = diff.get_val();
  if (w.rank == 0)
    printf("diff: %" PRId64 "\n", val);

  delete A;
}

int main(int argc, char** argv)
{
  int const in_num = argc;
  char** input_str = argv;
  int critter_mode=0;
  int b=BALL_SIZE;
  int d=2;

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
        if (b > A->nrow) b = A->nrow;
        // perturb(A);
#ifdef DEBUG
        if (w.rank == 0)
          printf("A:\n");
        A->print_matrix();
#endif
        if (!b)
          b = ceil(log2(A->nrow));

        Subemulator subem(A, b);
#if defined TEST || defined DEBUG
        Matrix<REAL> * H = subem.H;
        Vector<bpair> * q = subem.q;
#ifdef DEBUG
        if (w.rank == 0)
          printf("subemulator H:\n");
        H->print_matrix();
#endif
        // check number of edges
        int64_t A_nnz = A->nnz_tot;
        int64_t H_nnz = H->nnz_tot;
        // check distances are preserved for sampled vertices 
        Matrix<REAL> * A_dist = correct_dist(A, b);
        Matrix<REAL> * H_dist = correct_dist(H, b, "subemulator");
        int64_t A_dist_nnz = A_dist->nnz_tot;
        Matrix<REAL> * D = new Matrix<REAL>(A->nrow, A->ncol, A->symm|(A->is_sparse*SP), w, PLUS_TIMES_SR);
        Bivar_Function<REAL,REAL,REAL> distortion([](REAL x, REAL y){ return y > 0 ? x / y : 1; });
        distortion.intersect_only = true;
        (*D)["ij"] = distortion((*H_dist)["ij"], (*A_dist)["ij"]);
        double max_distort = D->norm_infty();
        Scalar<REAL> tot_distort(w);
        tot_distort[""] = Function<REAL,REAL>([](REAL x){ return x; })((*D)["ij"]);
        double avg_distort = tot_distort.get_val() / D->nnz_tot;
        // check distances are preserved for all vertices (including non-sampled)
        Matrix<REAL> * P = new Matrix<REAL>(A->nrow, A->ncol, A->symm|(A->is_sparse*SP), w, MIN_TIMES_SR);
        int64_t q_nprs;
        Pair<bpair> * q_prs;
        q->get_local_pairs(&q_nprs, &q_prs);
        Pair<REAL> p_prs[q_nprs];
        for (int64_t i = 0; i < q_nprs; ++i) {
          p_prs[i].k = q_prs[i].k + q_prs[i].d.vertex*A->nrow;
          p_prs[i].d = 1;
        }
        P->write(q_nprs, p_prs);
        Matrix<REAL> * E = new Matrix<REAL>(A->nrow, A->ncol, A->symm|(A->is_sparse*SP), w, MIN_TIMES_SR);
        Bivar_Function<REAL,REAL,REAL> mxm([](REAL x, REAL y){ return x*y; });
        mxm.intersect_only = true;
        (*E)["ij"] = mxm((*P)["ik"], (*H_dist)["kj"]);
        (*E)["ij"] = mxm((*E)["ik"], (*P)["jk"]);
        // (*E)["ij"] = (*P)["il"] * (*H_dist)["lk"] * (*P)["jk"]; // TODO: some floating point issues
        (*E)["ii"] = 0.0;
        Matrix<REAL> * F = new Matrix<REAL>(A->nrow, A->ncol, A->symm|(A->is_sparse*SP), w); // switch to (+, *) semiring
        (*F)["ij"] = (*E)["ij"];
        delete E;
        Transform<bpair,REAL>([](bpair pr, REAL & a){ a += pr.dist; })((*q)["i"], (*F)["ij"]);
        Transform<bpair,REAL>([](bpair pr, REAL & a){ a += pr.dist; })((*q)["j"], (*F)["ij"]);
        int64_t F_nnz = F->nnz_tot;
        (*D)["ij"] = distortion((*F)["ij"], (*A_dist)["ij"]);
        double max_distort_all = D->norm_infty();
        tot_distort[""] = Function<REAL,REAL>([](REAL x){ return x; })((*D)["ij"]);
        double avg_distort_all = tot_distort.get_val() / D->nnz_tot;
        delete F;
        delete P;
        delete H_dist;
        delete A_dist;
        delete D;
        if (w.rank == 0) {
          // check distances in subemulator
          printf("subemulator has %" PRId64 " nonzeros\n", H_nnz);
          int check = (int) (H_nnz <= A_nnz + A->nrow * b);
          if (check)
            printf("passed: subemulator has less than m+nb edges\n");
          else 
            printf("failed: subemulator has more than m+nb edges\n");

          printf("subemulator max distance distortion between sampled vertices is %f\n", max_distort);
          printf("subemulator avg distance distortion between sampled vertices is %f\n", avg_distort);
          check = (int) (max_distort >= 1 && max_distort <= 8);
          if (check)
            printf("passed: subemulator preserves distances among sampled vertices\n");
          else 
            printf("failed: subemulator does not preserve distances among sampled vertices\n");

          // check distances in correction of subemulator (ie subemulator + distances to leaders)
          printf("subemulator correction contains %" PRId64 " distances\n", F_nnz);
          if (F_nnz == A_dist_nnz)
            printf("passed: subemulator correction has same connectivity as input graph\n");
          else
            printf("failed: subemulator correction does not have the same connecitivity as input graph \n");

          printf("subemulator correction max distance distortion between all vertices is %f\n", max_distort_all);
          printf("subemulator correction avg distance distortion between all vertices is %f\n", avg_distort_all);
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
