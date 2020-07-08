#include "multigrid.h"

/**
 * \brief computes Multigrid for a 3D regular discretization
 */
int test_alg_multigrid(int64_t    N,
                       int        nlvl,
                       int *      nsm,
                       Matrix<REAL> & A,
                       Vector<REAL> & b,
                       Vector<REAL> & x_init,
                       Matrix<REAL> * P,
                       Matrix<REAL> * PTAP){

  Vector<REAL> x2(x_init);
  Timer_epoch vc("vcycle");
  vc.begin();
  double st_time = MPI_Wtime();
  vcycle(A, x_init, b, P, PTAP, N, nlvl, nsm);
  double vtime = MPI_Wtime()-st_time;
  vc.end();

  smooth_jacobi(A,x2,b,2*nsm[0]);
  Vector<REAL> r2(x2);
  r2["i"] = b["i"];
  r2["i"] -= A["ij"]*x2["j"];
  REAL rnorm_alt = r2.norm2();

  Vector<REAL> r(x_init);
  r["i"]  = b["i"];
  r["i"] -= A["ij"]*x_init["j"];
  REAL rnorm = r.norm2(); 
 
  bool pass = rnorm < rnorm_alt;

  if (A.wrld->rank == 0){
#ifndef TEST_SUITE
    printf("Algebraic multigrid with n %ld nlvl %d took %lf seconds, fine-grid only err = %E, multigrid err = %E\n",N,nlvl,vtime,rnorm_alt,rnorm); 
#endif
    if (pass) 
      printf("{ algebraic multigrid method } passed \n");
    else
      printf("{ algebraic multigrid method } failed \n");
  }
  return pass;

}

#ifndef TEST_SUITE
char* getCmdOption(char ** begin,
                   char ** end,
                   const   std::string & option){
  char ** itr = std::find(begin, end, option);
  if (itr != end && ++itr != end){
    return *itr;
  }
  return 0;
}


int main(int argc, char ** argv){
  int rank, np, pass, nlvl, ndiv, decay_exp, nsmooth, k;
  int * nsm;
  int64_t n;
  REAL sp_frac;
  int const in_num = argc;
  char ** input_str = argv;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  if (getCmdOption(input_str, input_str+in_num, "-n")){
    n = atoi(getCmdOption(input_str, input_str+in_num, "-n"));
    if (n < 0) n = 16;
  } else n = 16;

  if (getCmdOption(input_str, input_str+in_num, "-nlvl")){
    nlvl = atoi(getCmdOption(input_str, input_str+in_num, "-nlvl"));
    if (nlvl < 0) nlvl = 3;
  } else nlvl = 3;

  if (getCmdOption(input_str, input_str+in_num, "-ndiv")){
    ndiv = atoi(getCmdOption(input_str, input_str+in_num, "-ndiv"));
    if (ndiv < 0) ndiv = 2;
  } else ndiv = 2;

  if (getCmdOption(input_str, input_str+in_num, "-nsmooth")){
    nsmooth = atoi(getCmdOption(input_str, input_str+in_num, "-nsmooth"));
    if (nsmooth < 0) nsmooth = 3;
  } else nsmooth = 3;

  nsm = (int*)malloc(sizeof(int)*nlvl);
  std::fill(nsm, nsm+nlvl, nsmooth);

  char str[] = {'-','n','s','m','0','\0'};
  for (int i=0; i<nlvl; i++){
    str[4] = '0'+i;
    if (getCmdOption(input_str, input_str+in_num, str)){
      int insm = atoi(getCmdOption(input_str, input_str+in_num, str));
      if (insm > 0) nsm[i] = insm;
    }
  }

  if (getCmdOption(input_str, input_str+in_num, "-decay_exp")){
    decay_exp = atoi(getCmdOption(input_str, input_str+in_num, "-decay_exp"));
    if (decay_exp < 0) decay_exp = 3;
  } else decay_exp = 3;
  
  if (getCmdOption(input_str, input_str+in_num, "-sp_frac")){
    sp_frac = atof(getCmdOption(input_str, input_str+in_num, "-sp_frac"));
    if (sp_frac < 0) sp_frac = .01;
  } else sp_frac = .01;

  if (getCmdOption(input_str, input_str+in_num, "-k")) {
    k = atoi(getCmdOption(input_str, input_str+in_num, "-k"));
    if (k < 0) k = 5;
  } else k = -1;
  // K13 : 1594323 (matrix size)
  // K6 : 729; 531441 vertices
  // k5 : 243
  // k7 : 2187
  // k8 : 6561
  // k9 : 19683

  nlvl--;
  int64_t all_lvl_ndiv=1;
  for (int i=0; i<nlvl; i++){ all_lvl_ndiv *= ndiv; }

  assert(n%all_lvl_ndiv == 0);

  {
    World dw(argc, argv);

    if (rank == 0){
      printf("Running algebraic smoothed multigrid method with %d levels with divisor %d in V-cycle, matrix dimension %ld, %d smooth iterations, decayed based on 3D indexing with decay exponent of %d\n",nlvl,ndiv,n,nsmooth, decay_exp);
      printf("number of smoothing iterations per level is ");
      for (int i=0; i<nlvl+1; i++){ printf("%d ",nsm[i]); }
      printf("\n");
    }
    Matrix<REAL> A;
    Matrix<REAL> * P;
    Matrix<REAL> * PTAP;
    Vector<REAL> b(n*n*n,dw,"b");
    Vector<REAL> x(n*n*n,dw,"x");

    if (k != -1) {
      int64_t matSize = pow(3, k);
      auto B = generate_kronecker(&dw, k);

      // L = D - A // TODO: laplacian matrix is different on multiple processes
      Function<wht, wht> addinv([](wht a){ return -a; });
      (*B)["ij"] = addinv((*B)["ij"]);
      (*B)["ii"] = (*B)["ij"];
      (*B)["ii"] = addinv((*B)["ii"]);

      if (dw.rank == 0) {
        printf("Running multigrid on Kronecker graph K: %d matSize: %ld\n", k, matSize);
      }
      B->print_matrix();
      delete B;

      setup_laplacian(n, nlvl, sp_frac, ndiv, decay_exp, A, P, PTAP, dw);
      b.fill_random(-1.E-1, 1.E-1);
    } else {
      if (dw.rank == 0) {
        printf("No graph specified\n");
      }
    }
    pass = test_alg_multigrid(n*n*n, nlvl, nsm, A, b, x, P, PTAP);
   // assert(pass);
  }

  MPI_Finalize();
  return 0;
}

#endif
