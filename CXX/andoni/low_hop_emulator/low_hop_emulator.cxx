#include "low_hop_emulator.h"

LowHopEmulator::LowHopEmulator(Matrix<REAL> * A, int b, int mode_) {
  mode = mode_;
  hierarchy(A, b);
  if (mode)
    collapse(A);
}

LowHopEmulator::~LowHopEmulator() {
  // for (int i = 0; i < NUM_LEVELS; ++i)
  //   delete subems[i];
  // if (mode)
  //   delete G;
}

void LowHopEmulator::hierarchy(Matrix<REAL> * A, int b) {
  // bottom level
  int n = A->nrow;
  Monoid<bpair> bpair_monoid = get_bpair_monoid();
  Vector<bpair> * q = new Vector<bpair>(n, *A->wrld, bpair_monoid);
  Vector<int> ID = arange(0, n, 1, *A->wrld);
  (*q)["i"] = Function<int,bpair>([](int id){ return bpair(id, 0.0); })(ID["i"]);
  subems[0] = new Subemulator(A, q, b);

  for (int i = 1; i < NUM_LEVELS; ++i) { // TODO: stop if 0 vertices
    subems[i] = new Subemulator(subems[i-1]->H, b);
    // b = pow(b, 1.25); // TODO: implement
  }
}

void LowHopEmulator::collapse(Matrix<REAL> * A) { // A is passed for its metadata
  G = new Matrix<REAL>(A->nrow, A->ncol, A->symm|(A->is_sparse*SP), *A->wrld, MIN_PLUS_SR);
  for (int i = NUM_LEVELS-1; i >= 0; --i) { // collapse sparsest levels first
    (*G)["ij"] += pow(PENALTY, (NUM_LEVELS-1) - i) * (*subems[i]->H)["ij"]; // TODO: is penalty correct?
  }
}

Matrix<REAL> * LowHopEmulator::sssp_hierarchy(int vertex) {
  return NULL;
}

Matrix<REAL> * LowHopEmulator::sssp_collapse(int vertex) {
  return NULL;
}

Matrix<REAL> * LowHopEmulator::sssp(int vertex) {
  if (mode)
    return sssp_collapse(vertex);
  else
    return sssp_hierarchy(vertex);
}

// all pairs shortest path on collapsed low hop emulator
Matrix<REAL> * LowHopEmulator::apsp() {
  Matrix<REAL> * D = new Matrix<REAL>(*G);
  Matrix<REAL> * D_prev = new Matrix<REAL>(D->nrow, D->ncol, D->symm|(D->is_sparse*SP), *D->wrld, MIN_PLUS_SR);
  int hops = 0;
  while (are_matrices_different(D, D_prev)) {
    (*D_prev)["ij"] = (*D)["ij"];
    (*D)["ij"] += (*G)["ik"] * (*D)["kj"];
    ++hops;
  }
  if (D->wrld->rank == 0)
    printf("low hop emulator has %d hops\n", hops);
  delete D_prev;
  return D;
}
