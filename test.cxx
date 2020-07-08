#include "multigrid.h"

char* getCmdOption(char ** begin,
                   char ** end,
                   const   std::string & option) {
  char ** itr = std::find(begin, end, option);
  if (itr != end && ++itr != end){
    return *itr;
  }
  return 0;
}

int main(int argc, char** argv)
{
  int rank;
  int np;
  int const in_num = argc;
  char** input_str = argv;

  int k;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  {
    World w(argc, argv);
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
    
    if (k != -1) {
      int64_t matSize = pow(3, k);
      auto B = generate_kronecker(&w, k);

      // L = D - A
      Function<wht, wht> addinv([](wht a){ return -a; });
      (*B)["ij"] = addinv((*B)["ij"]);
      (*B)["ii"] = (*B)["ij"];
      (*B)["ii"] = addinv((*B)["ii"]);

      if (w.rank == 0) {
        printf("Running connectivity on Kronecker graph K: %d matSize: %ld\n", k, matSize);
      }
      B->print_matrix();
      delete B;
    }
    else {
      if (w.rank == 0) {
        printf("No graph specified\n");
      }
    }
  }
  MPI_Finalize();
  return 0;
}
