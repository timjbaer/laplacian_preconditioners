#include "low_hop_emulator.h"

Subemulator::Subemulator(Matrix<REAL> * A, int b) { // A is on (min, +)
  assert(A->is_sparse); // not strictly necessary, but much more efficient
  B = ball_matmat(A, b);
  Vector<int> * S = samples(A, b);
  connects(A, S, b);
}

Subemulator::~Subemulator() {
  delete H;
  delete q;
  delete B;
}

Vector<int> * Subemulator::samples(Matrix<REAL> * A, int b) { // FIXME: vertex 0 is never selected
  int n = A->nrow;
  World * w = A->wrld;
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
  (*S)["i"] = (*S)["i"] * ID["i"];
  S->sparsify(); // TODO: sparsify earlier?
  // S->print();
  delete is_close;
  return S;
}

void Subemulator::connects(Matrix<REAL> * A, Vector<int> * S, int b) {
  int n = A->nrow;
  World * w = A->wrld;

  // if (w->rank == 0) printf("S\n");
  // S->print();

  // if (w->rank == 0) printf("B\n");
  // B->print_matrix();

  Monoid<bpair> bpair_monoid = get_bpair_monoid();
  q = new Vector<bpair>(n, *w, bpair_monoid);
  Bivar_Function<REAL,int,bpair> f([](REAL a, int s){ return bpair(s, a); });
  f.intersect_only = true;
  (*q)["i"] = f((*B)["ij"], (*S)["j"]);
  // if (w->rank == 0) printf("q\n");
  // q->print();

  // Matrix<REAL> * C = new Matrix<REAL>(n, n, A->symm|(A->is_sparse*SP), *w, MIN_PLUS_SR);
  Matrix<REAL> * C = new Matrix<REAL>(n, n, A->symm|(A->is_sparse*SP), *w, MAX_MONOID); // MAX_MONOID is a hack to avoid Transform accumulating
  (*C)["ij"] = (*A)["ij"];
  // std::function<REAL(bpair,REAL,bpair)> f = [](bpair x, REAL a, bpair y){ return x.dist + a + y.dist; };
  // Tensor<bpair> * vec_list[2] = {q, q};
  // Multilinear<REAL,bpair,REAL>(A, vec_list, C, f); // e = (q_i,q_j) for (i,j) \in E
  // Multilinear<REAL,bpair,REAL>(B, vec_list, C, f); // e = (q_i,q_j) for j \in B_i
  Transform<bpair,REAL>([](bpair pr, REAL & c){ c += pr.dist; })((*q)["i"], (*C)["ij"]);
  Transform<bpair,REAL>([](bpair pr, REAL & c){ c += pr.dist; })((*q)["j"], (*C)["ij"]);
  // if (w->rank == 0) printf("C:\n");
  // C->print_matrix();
  Vector<int> * q_vertices = new Vector<int>(n, *w);
  (*q_vertices)["i"] = Function<bpair,int>([](bpair pair){ return pair.vertex; })((*q)["i"]);
  H = PTAP<REAL>(C, q_vertices); // FIXME: put H on (min, +)
  delete q_vertices;
  delete C;
}
