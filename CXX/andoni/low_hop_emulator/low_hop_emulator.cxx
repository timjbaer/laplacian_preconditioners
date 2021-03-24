#include "low_hop_emulator.h"

Matrix<REAL> * collapse(Matrix<REAL> * A, int b) {
  Matrix<REAL> * G = new Matrix<REAL>(A->nrow, A->ncol, A->symm|(A->is_sparse*SP), *A->wrld, MIN_PLUS_SR);
  Subemulator * subem;
  Matrix<REAL> * H = new Matrix<REAL>(*A);
  for (int i = 0; i < NUM_LEVELS; ++i) {
    subem = new Subemulator(H, b);
    delete H;
    H = new Matrix<REAL>(*subem->H); // TODO: avoid copy, manipulate pointers
    (*G)["ij"] += pow(PENALTY, NUM_LEVELS - i) * (*H)["ij"];
    b = pow(b, 1.25); 
  }
  delete H;
  return G;
}

Matrix<REAL> * hierarchy(Matrix<REAL> * A, int b) {
}

LowHopEmulator::LowHopEmulator(Matrix<REAL> * A, int b, int mode) {
  if (mode)
    G = hierarchy(A, b);
  else
    G = collapse(A, b);
}
