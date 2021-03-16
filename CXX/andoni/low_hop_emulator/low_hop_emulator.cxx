#include "low_hop_emulator.h"

Subemulator::Subemulator(Matrix<REAL> * A, int b) { // A is on (min, +)
//  int n = A->nrow;
//  H = new Matrix<REAL>();
//  q = new Vector<int>();
//  Vector<int> * S = samples(A, b);
//  connects(A, S, b);
}

Subemulator::~Subemulator() {
  delete H;
}

Vector<int> * Subemulator::samples(Matrix<REAL> * A, int b) {
//   Vector<int> * S = new Vector<int>(); // put on min plus
//   Vector<int> ID = arange();
//   (*S)["i"] = Function<int>([](int id){ return id : INT_MAX ? (double)rand()/RAND_MAX < 0.5; })(ID["i"]);
//   Matrix<REAL> * B = ball();
//   (*S)["i"] += Function<REAL,int,int>([](int a, int id){ return id : INT_MAX ? a > 0})((*B)["ij"], ID["j"]);
//   return S;
}

void Subemulator::connects(Matrix<REAL> * A, Vector<int> * S, int b) {
//   (*q)["i"] = Function<REAL,int,int>([](REAL ){ })();
}
