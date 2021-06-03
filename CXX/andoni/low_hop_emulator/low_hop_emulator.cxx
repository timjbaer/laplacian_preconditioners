#include "low_hop_emulator.h"

LowHopEmulator::LowHopEmulator(Matrix<REAL> * A, int b0_, int mode_) {
  n = A->nrow;
  b0 = b0_;
  w = new World(*A->wrld);
  mode = mode_;
  hierarchy(A);
  if (mode)
    collapse(A);
}

LowHopEmulator::~LowHopEmulator() {
  // for (int i = 0; i < NUM_LEVELS; ++i)
  //   delete subems[i];
  // if (mode)
  //   delete G;
}

void LowHopEmulator::hierarchy(Matrix<REAL> * A) {
  int b = b0;
  Vector<bpair> * q = NULL;
  subems[0] = new Subemulator(A, q, b);
  // if (w->rank == 0) printf("ball of input graph\n");
  // subems[0]->B->print_matrix();
  for (int i = 1; i <= NUM_LEVELS; ++i) {
    subems[i] = new Subemulator(subems[i-1], b);
    // if (w->rank == 0) printf("subemulator:\n");
    // subems[i]->H->print_matrix();
    // b = pow(b, 1.25); // TODO: implement
  }
}

void LowHopEmulator::collapse(Matrix<REAL> * A) { // A is passed for its metadata
  G = new Matrix<REAL>(n, n, A->symm|(A->is_sparse*SP), *w, MIN_PLUS_SR);
  (*G)["ii"] = 0.0;
  for (int i = NUM_LEVELS; i >= 0; --i) { // collapse sparsest levels first
    (*G)["ij"] += (*(subems[i]->B))["ij"]; // TODO: add penalty
    // (*G)["ij"] += (*(subems[i]->H))["ij"]; // TODO: add penalty
  }
  (*G)["ij"] += (*G)["ji"]; // make undirected
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
  assert(mode == 1);
  Matrix<REAL> * D = new Matrix<REAL>(*G);
  Matrix<REAL> * D_prev = new Matrix<REAL>(n, n, D->symm|(D->is_sparse*SP), *w, MIN_PLUS_SR);
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
