/*
 * https://github.com/cyclops-community/ctf/blob/master/examples/algebraic_multigrid.cxx
 */

#include "multigrid.h"

#define ERR_REPORT

void smooth_jacobi(Matrix<REAL> & A, Vector<REAL> & x, Vector <REAL> & b, int nsm){
  Timer jacobi("jacobi");
  Timer jacobi_spmv("jacobi_spmv");

  jacobi.start();
  Vector<REAL> d(x.len, *x.wrld);
  d["i"] = A["ii"];
  Transform<REAL>([](REAL & d){ d= fabs(d) > 0.0 ? 1./d : 0.0; })(d["i"]);
  Matrix<REAL> R(A);
  R["ii"] = 0.0;
  Vector<REAL> x1(x.len, *x.wrld);
 
  double omega = .333;
  //20 iterations of Jacobi, should probably be a parameter or some convergence check instead
  for (int i=0; i<nsm; i++){
    jacobi_spmv.start();
    x1["i"] = -1.*R["ij"]*x["j"];
    jacobi_spmv.stop();
    x1["i"] += b["i"];
    x1["i"] *= d["i"];
    x["i"] *= (1.-omega);
    x["i"] += omega*x1["i"];
    //x["i"] = x1["i"];
#ifdef ERR_REPORT
    Vector<REAL> r(b);
    r["i"] -= A["ij"]*x["j"];
    r.print();
    REAL rnorm = r.norm2();
    if (A.wrld->rank == 0) printf("r norm is %E\n",rnorm);
#endif
  }
  jacobi.stop();
}

void vcycle(Matrix<REAL> & A, Vector<REAL> & x, Vector<REAL> & b, Matrix<REAL> * P, Matrix<REAL> * PTAP, int64_t N, int nlevel, int * nsm){
  //do smoothing using Jacobi
  char tlvl_name[] = {'l','v','l',(char)('0'+nlevel),'\0'};
  Timer tlvl(tlvl_name);
  tlvl.start();
  Vector<REAL> r(N,*A.wrld,"r");
#ifdef ERR_REPORT
  r["i"] -= A["ij"]*x["j"];
  r["i"] += b["i"];
  REAL rnorm0 = r.norm2();
#endif
#ifdef ERR_REPORT
  if (A.wrld->rank == 0) printf("At level %d residual norm was %1.2E initially\n",nlevel,rnorm0);
#endif
  if (N==1){
    x["i"] = Function<REAL>([](REAL a, REAL b){ return b/a; })(A["ij"],b["j"]);
  } else {
    smooth_jacobi(A,x,b,nsm[0]);
  }
  r["i"] = b["i"];
  r["i"] -= A["ij"]*x["j"];
#ifdef ERR_REPORT
  REAL rnorm = r.norm2();
#endif
  if (nlevel == 0){
#ifdef ERR_REPORT
    if (A.wrld->rank == 0) printf("At level %d (coarsest level), residual norm was %1.2E initially\n",nlevel,rnorm0);
    if (A.wrld->rank == 0) printf("At level %d (coarsest level), residual norm was %1.2E after smooth\n",nlevel,rnorm);
#endif
    return; 
  }
  int64_t m = P[0].lens[1];

  //smooth the restriction/interpolation operator P = (I-omega*diag(A)^{-1}*A)T
  Timer rstr("restriction");
  rstr.start();

  //restrict residual vector
  Vector<REAL> PTr(m, *x.wrld);
  PTr["i"] += P[0]["ji"]*r["j"];
 
  //coarses initial guess should be zeros
  Vector<REAL> zx(m, *b.wrld);
  rstr.stop(); 
  tlvl.stop();
  //recurse into coarser level
  vcycle(PTAP[0], zx, PTr, P+1, PTAP+1, m, nlevel-1, nsm+1);
  tlvl.start();

  //interpolate solution to residual equation at coraser level back
  x["i"] += P[0]["ij"]*zx["j"]; 
 
#ifdef ERR_REPORT
  r["i"] = b["i"];
  r["i"] -= A["ij"]*x["j"];
  REAL rnorm2 = r.norm2();
#endif
  //smooth new solution
  smooth_jacobi(A,x,b,nsm[0]);
  tlvl.stop();
#ifdef ERR_REPORT
  r["i"] = b["i"];
  r["i"] -= A["ij"]*x["j"];
  REAL rnorm3 = r.norm2();
  if (A.wrld->rank == 0) printf("At level %d, residual norm was %1.2E initially\n",nlevel,rnorm0);
  if (x.wrld->rank == 0) printf("At level %d, n=%ld residual norm was %1.2E after initial smooth\n",nlevel,N,rnorm);
  if (A.wrld->rank == 0) printf("At level %d, residual norm was %1.2E after coarse recursion\n",nlevel,rnorm2);
  if (A.wrld->rank == 0) printf("At level %d, residual norm was %1.2E after final smooth\n",nlevel,rnorm3);
#endif
}

void setup(Matrix<REAL> & A, Matrix<REAL> * T, int N, int nlevel, Matrix<REAL> * P, Matrix<REAL> * PTAP){
  if (nlevel == 0) return;

  char slvl_name[] = {'s','l','v','l',(char)('0'+nlevel),'\0'};
  Timer slvl(slvl_name);
  slvl.start();
  int64_t m = T[0].lens[1];
  P[0] = Matrix<REAL>(N, m, SP, *T[0].wrld);
  Matrix<REAL> D(N,N,SP,*A.wrld);
  D["ii"] = A["ii"];
  REAL omega=.333;
  Transform<REAL>([=](REAL & d){ d= omega/d; })(D["ii"]);
  Timer trip("triple_matrix_product_to_form_T");
  trip.start();
  Matrix<REAL> F(P[0]);
  F["ik"] = A["ij"]*T[0]["jk"];
  P[0]["ij"] = T[0]["ij"];
  P[0]["ik"] -= D["il"]*F["lk"];
  trip.stop();
  
  int atr = 0;
  if (A.is_sparse){ 
    atr = atr | SP;
  }
  Matrix<REAL> AP(N, m, atr, *A.wrld);
  PTAP[0] = Matrix<REAL>(m, m, atr, *A.wrld);
 
  Timer trip2("triple_matrix_product_to_form_PTAP");
  trip2.start();
  //restrict A via triple matrix product, should probably be done outside v-cycle
  AP["lj"] = A["lk"]*P[0]["kj"];
  PTAP[0]["ij"] = P[0]["li"]*AP["lj"];

  trip2.stop();
  slvl.stop();
  setup(PTAP[0], T+1, m, nlevel-1, P+1, PTAP+1);
}

void setup_laplacian(int64_t         n,
                     int             nlvl,
                     REAL            sp_frac,
                     int             ndiv,
                     int             decay_exp,
                     Matrix<REAL>  & A,
                     Matrix<REAL> *& P,
                     Matrix<REAL> *& PTAP,
                     World &         dw){
  int64_t n3 = n*n*n;
  Timer tct("initialization");
  tct.start();
  A = Matrix<REAL>(n3, n3, SP, dw);
  srand48(dw.rank*12);
  A.fill_sp_random(0.0, 1.0, sp_frac);

  A["ij"] += A["ji"];
  REAL pn = sqrt((REAL)n);
  A["ii"] += pn;

  if (dw.rank == 0){
    printf("Generated matrix with dimension %1.2E and %1.2E nonzeros\n", (REAL)n3, (REAL)A.nnz_tot);
    fflush(stdout);
  }

  Matrix<std::pair<REAL, int64_t>> B(n3,n3,SP,dw,Set<std::pair<REAL, int64_t>>());

  int64_t * inds;
  REAL * vals;
  std::pair<REAL,int64_t> * new_vals;
  int64_t nvals;
  A.get_local_data(&nvals, &inds, &vals, true);

  new_vals = (std::pair<REAL,int64_t>*)malloc(sizeof(std::pair<REAL,int64_t>)*nvals);

  for (int64_t i=0; i<nvals; i++){
    new_vals[i] = std::pair<REAL,int64_t>(vals[i],abs((inds[i]%n3) - (inds[i]/n3)));
  }

  B.write(nvals,inds,new_vals);
  delete [] vals;
  free(new_vals);
  free(inds);

  Transform< std::pair<REAL,int64_t> >([=](std::pair<REAL,int64_t> & d){ 
    int64_t x =  d.second % n;
    int64_t y = (d.second / n) % n;
    int64_t z =  d.second / n  / n;
    if (x+y+z > 0)
      d.first = d.first/pow((REAL)(x+y+z),decay_exp/2.);
    }
  )(B["ij"]);
  
  A["ij"] = Function< std::pair<REAL,int64_t>, REAL >([](std::pair<REAL,int64_t> p){ return p.first; })(B["ij"]);

  Matrix<REAL> * T = new Matrix<REAL>[nlvl];
  int64_t m=n3;
  int tot_ndiv = ndiv*ndiv*ndiv;
  for (int i=0; i<nlvl; i++){
    int64_t m2 = m/tot_ndiv;
    T[i] = Matrix<REAL>(m, m2, SP, dw);
    int64_t mmy = m2/dw.np;
    if (dw.rank < m2%dw.np) mmy++;
    Pair<REAL> * pairs = (Pair<REAL>*)malloc(sizeof(Pair<REAL>)*mmy*tot_ndiv);
    int64_t nel = 0;
    for (int64_t j=dw.rank; j<m2; j+=dw.np){
      for (int k=0; k<tot_ndiv; k++){
        pairs[nel] = Pair<REAL>(j*m+j*tot_ndiv+k, 1.0);
        nel++;
      }
    }
    T[i].write(nel, pairs);
    delete [] pairs;
    m = m2;
  }
  tct.stop();

  P = new Matrix<REAL>[nlvl];
  PTAP = new Matrix<REAL>[nlvl];

  Timer_epoch ve("setup");
  ve.begin();
  setup(A, T, n3, nlvl, P, PTAP);
  ve.end();
}
