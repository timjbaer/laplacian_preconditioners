#ifndef __LOW_HOP_EMULATOR_H__
#define __LOW_HOP_EMULATOR_H__

#include "ball.h"

class Subemulator {
  public:
    Matrix<REAL> * H; // subemulator
    Vector<bpair> * q; // leaders
    Matrix<REAL> * B; // ball

    Subemulator(Matrix<REAL> * A, int b);

    ~Subemulator();

    Vector<int> * samples(Matrix<REAL> * A, int b);

    void connects(Matrix<REAL> * A, Vector<int> * S, int b);
};

class DistOracle {

};

class LowHopEm {

};

static Semiring<int> MAX_TIMES_SR(0,
    [](int a, int b) {
      return std::max(a, b);
    },
    MPI_MAX,
    1,
    [](int a, int b) {
      return a * b;
    });

static Monoid<REAL> MAX_MONOID(0,
    [](REAL a, REAL b) {
      return std::max(a,b);
    },
    MPI_MAX
    );

// return B where B[i,j] = A[p[i],p[j]], or if P is P[i,j] = p[i], compute B = P^T A P
template<typename T>
Matrix<T>* PTAP(Matrix<T>* A, Vector<int>* p){
  Timer t_ptap("ptap");
  t_ptap.start();
  int np = p->wrld->np;
  int64_t n = p->len;
  Pair<int> * pprs;
  int64_t npprs;
  //get local part of p
  p->get_local_pairs(&npprs, &pprs);
  assert((npprs <= (n+np-1)/np) && (npprs >= (n/np)));
  assert(A->ncol == n);
  assert(A->nrow == n);
  Pair<T> * A_prs;
  int64_t nprs;
  {
    //map matrix so rows are distributed as elements of p, ensures for each element of p, this process also owns the row of A (A1)
    Matrix<T> A1(n, n, "ij", Partition(1,&np)["i"], Idx_Partition(), SP*(A->is_sparse), *A->wrld, *A->sr);
    A1["ij"] = A->operator[]("ij");
    A1.get_local_pairs(&nprs, &A_prs, true);
    //use fact p and rows of A are distributed cyclically, to compute P^T * A
    for (int64_t i=0; i<nprs; i++){
      A_prs[i].k = (A_prs[i].k/n)*n + pprs[(A_prs[i].k%n)/np].d;
    }
  }
  {
    //map matrix so rows are distributed as elements of p, ensures for each element of p, this process also owns the column of A (A1)
    Matrix<T> A2(n, n, "ij", Partition(1,&np)["j"], Idx_Partition(), SP*(A->is_sparse), *A->wrld, *A->sr);
    //write in P^T A into A2
    A2.write(nprs, A_prs);
    delete [] A_prs;
    A2.get_local_pairs(&nprs, &A_prs, true);
    //use fact p and cols of A are distributed cyclically, to compute P^T A * P
    for (int64_t i=0; i<nprs; i++){
      A_prs[i].k = (A_prs[i].k%n) + pprs[(A_prs[i].k/n)/np].d*n;
    }
  }
  Matrix<T> * PTAP = new Matrix<T>(n, n, SP*(A->is_sparse), *A->wrld, MIN_PLUS_SR); // FIXME: somewhat hacky
  PTAP->write(nprs, A_prs);
  delete [] A_prs;
  t_ptap.stop();
  return PTAP;
}
template Matrix<REAL>* PTAP<REAL>(Matrix<REAL>* A, Vector<int>* p);


#endif
