#include "subemulator.h"

Subemulator::Subemulator(Matrix<REAL> * A, int b_) { // A is on (min, +)
  Timer t_subemulator("subemulator");
  t_subemulator.start();
  assert(A->is_sparse); // not strictly necessary, but much more efficient
  b = b_;
  // B = ball_matmat(A, b);
  B = new Matrix<REAL>(A->nrow, A->ncol, A->symm|(A->is_sparse*SP), *A->wrld, *A->sr);
  Vector<bvector<BALL_SIZE>> * ball = ball_bvector<BALL_SIZE>(A, 0, 0);
  bvec_to_mat(B, ball);
  (*B)["ii"] = MAX_REAL; // ball may contain itself as a closest neighbor
  delete ball;
  assert(B->is_sparse); // not strictly necessary, but much more efficient
  assert(B->topo->order == 2 || B->wrld->np < 4); // A distributed on 2D processor grid (not strictly necessary but scales better)
  Vector<int> * S = samples();
  connects(A, S);
  t_subemulator.stop();
}

Subemulator::~Subemulator() {
  delete H;
  delete q;
  delete B;
}

Vector<int> * Subemulator::samples() {
  Timer t_samples("samples");
  t_samples.start();
  int n = B->nrow;
  World * w = B->wrld;
  // if (w->rank == 0) printf("B\n");
  // B->print_matrix();
  Vector<int> * S = new Vector<int>(n, *w, MAX_TIMES_SR);
  Vector<int> ID = arange(0, n, 1, *w);
  (*S)["i"] = Function<int,int>([](int id){ return (int) ((double)rand()/RAND_MAX < 0.5); })(ID["i"]); // ID is not actually needed here
  // S->print();
  Bivar_Function<REAL,int,int> f([](REAL b, int s){ return s; }); // alternative to making S sparse and returning 1
  f.intersect_only = true;
  Vector<int> * is_close = new Vector<int>(n, *w, MAX_TIMES_SR);
  (*is_close)["i"] = f((*B)["ij"],(*S)["j"]);
  // if (w->rank == 0) printf("is_close\n");
  // is_close->print();
  (*S)["i"] += Function<int,int>([](int id){ return id ^ 1; })((*is_close)["i"]);
  (*S)["i"] = (*S)["i"] * ID["i"]; // FIXME: key-value pair (0,0) is always written but since value 0 is addid, vertex 0 is never selected as a leader
  S->sparsify(); // TODO: sparsify earlier?
  // if (w->rank == 0) printf("S\n");
  // S->print();
  delete is_close;
  t_samples.stop();
  int64_t S_nnz = S->nnz_tot;
  if (w->rank == 0) {
    printf("subemulator has %d vertices\n", S_nnz);
    int check = (int) (S_nnz <= 0.75*n);
    if (check)
      printf("passed: subemulator has less than 0.75n vertices\n");
    else 
      printf("failed: subemulator has more than 0.75n vertices\n");
  }
  return S;
}

// return B where B[i,j] = A[p[i],p[j]], or if P is P[i,j] = p[i], compute B = P^T A P
// specialized implementation, see mst for templated implementation
Matrix<REAL>* PTAP(Matrix<REAL>* A, Vector<bpair>* p){
  assert(A->topo->order == 2 || A->wrld->np < 4); // A distributed on 2D processor grid (not strictly necessary but scales better)
  Timer t_ptap("ptap");
  t_ptap.start();
  int np = p->wrld->np;
  int64_t n = p->len;
  Pair<bpair> * pprs;
  int64_t npprs;
  //get local part of p
  p->get_local_pairs(&npprs, &pprs);
  assert((npprs <= (n+np-1)/np) && (npprs >= (n/np)));
  assert(A->ncol == n);
  assert(A->nrow == n);
  Pair<REAL> * A_prs;
  int64_t nprs;
  {
    //map matrix so rows are distributed as elements of p, ensures for each element of p, this process also owns the row of A (A1)
    Matrix<REAL> A1(n, n, "ij", Partition(1,&np)["i"], Idx_Partition(), SP*(A->is_sparse), *A->wrld, *A->sr);
    A1["ij"] = A->operator[]("ij");
    A1.get_local_pairs(&nprs, &A_prs, true);
    //use fact p and rows of A are distributed cyclically, to compute P^T * A
    for (int64_t i=0; i<nprs; i++){
      A_prs[i].k = (A_prs[i].k/n)*n + pprs[(A_prs[i].k%n)/np].d.vertex;
      A_prs[i].d += pprs[(A_prs[i].k%n)/np].d.dist;
    }
  }
  {
    //map matrix so rows are distributed as elements of p, ensures for each element of p, this process also owns the column of A (A1)
    Matrix<REAL> A2(n, n, "ij", Partition(1,&np)["j"], Idx_Partition(), SP*(A->is_sparse), *A->wrld, *A->sr);
    //write in P^T A into A2
    A2.write(nprs, A_prs);
    delete [] A_prs;
    A2.get_local_pairs(&nprs, &A_prs, true);
    //use fact p and cols of A are distributed cyclically, to compute P^T A * P
    for (int64_t i=0; i<nprs; i++){
      A_prs[i].k = (A_prs[i].k%n) + pprs[(A_prs[i].k/n)/np].d.vertex*n;
      A_prs[i].d += pprs[(A_prs[i].k/n)/np].d.dist;
    }
  }
  Matrix<REAL> * PTAP = new Matrix<REAL>(n, n, SP*(A->is_sparse), *A->wrld, MIN_PLUS_SR);
  PTAP->write(nprs, A_prs);
  delete [] A_prs;
  t_ptap.stop();
  return PTAP;
}

void Subemulator::connects(Matrix<REAL> * A, Vector<int> * S) {
  Timer t_connects("connects");
  t_connects.start();
  int n = A->nrow;
  World * w = A->wrld;

  if (w->rank == 0) printf("B\n");
  B->print_matrix();

  Monoid<bpair> bpair_monoid = get_bpair_monoid();
  q = new Vector<bpair>(n, *w, bpair_monoid);
  Bivar_Function<REAL,int,bpair> f([](REAL a, int s){ return bpair(s, a); });
  f.intersect_only = true;
  (*q)["i"] = f((*B)["ij"], (*S)["j"]);

  if (w->rank == 0) printf("q\n");
  q->print();

  Matrix<REAL> * C = new Matrix<REAL>(n, n, A->symm|(A->is_sparse*SP), *w, MIN_PLUS_SR);
  (*C)["ij"] = (*A)["ij"] + (*B)["ij"];
  H = PTAP(C, q);
  delete C;
  t_connects.stop();
}

// void Subemulator::connects(Matrix<REAL> * A, Vector<int> * S) {
//   Timer t_connects("connects");
//   t_connects.start();
//   int n = A->nrow;
//   World * w = A->wrld;
// 
//   // if (w->rank == 0) printf("S\n");
//   // S->print();
// 
//   if (w->rank == 0) printf("B\n");
//   B->print_matrix();
// 
//   Monoid<bpair> bpair_monoid = get_bpair_monoid();
//   q = new Vector<bpair>(n, *w, bpair_monoid);
//   Bivar_Function<REAL,int,bpair> f([](REAL a, int s){ return bpair(s, a); });
//   f.intersect_only = true;
//   (*q)["i"] = f((*B)["ij"], (*S)["j"]);
//   if (w->rank == 0) printf("q\n");
//   q->print();
// 
//   // Matrix<REAL> * C = new Matrix<REAL>(n, n, A->symm|(A->is_sparse*SP), *w, MIN_PLUS_SR);
//   // std::function<REAL(bpair,REAL,bpair)> f = [](bpair x, REAL a, bpair y){ return x.dist + a + y.dist; };
//   // Tensor<bpair> * vec_list[2] = {q, q};
//   // Multilinear<REAL,bpair,REAL>(A, vec_list, C, f); // e = (q_i,q_j) for (i,j) \in E
//   // Multilinear<REAL,bpair,REAL>(B, vec_list, C, f); // e = (q_i,q_j) for j \in B_i
//   // Matrix<REAL> * C = new Matrix<REAL>(n, n, A->symm|(A->is_sparse*SP), *w, MAX_MONOID); // MAX_MONOID is a hack to avoid Transform accumulating
//   Matrix<REAL> * C = new Matrix<REAL>(n, n, A->symm|(A->is_sparse*SP), *w); // MAX_MONOID is a hack to avoid Transform accumulating
//   // (*C)["ij"] = (*A)["ij"];
//   // Bivar_Function<REAL,REAL,REAL> sum([](REAL a, REAL b){ return a + b; });
//   // sum.intersect_only = true;
//   // (*C)["ij"] = sum((*A)["ij"], (*B)["ij"]);
//   (*C)["ij"] = (*A)["ij"] + (*B)["ij"];
//   // Transform<bpair,REAL>([](bpair pr, REAL & c){ c += pr.dist; })((*q)["i"], (*C)["ij"]);
//   // Bivar_Function<REAL,bpair,REAL> f2([](REAL a, bpair pr){ return a + pr.dist; });
//   // f2.intersect_only = true;
//   // (*C)["ij"] = f2((*A)["ij"], (*q)["i"]);
//   // Transform<bpair,REAL>([](bpair pr, REAL & c){ c += pr.dist; })((*q)["j"], (*C)["ij"]);
//   // if (w->rank == 0) printf("C:\n");
//   // C->print_matrix();
//   // Vector<int> * q_vertices = new Vector<int>(n, *w);
//   // (*q_vertices)["i"] = Function<bpair,int>([](bpair pair){ return std::max(pair.vertex, 0); })((*q)["i"]); // FIXME: if vertex 0 is a leader, PTAP is not correct
//   // H = PTAP(C, q_vertices);
//   H = PTAP(C, q);
//   // delete q_vertices;
//   delete C;
//   t_connects.stop();
// }
