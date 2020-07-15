#include "graph.h"

Int64Pair::Int64Pair(int64_t i1, int64_t i2) {
  this->i1 = i1;
  this->i2 = i2;
}

Int64Pair Int64Pair::swap() {
  return {this->i2, this->i1};
}

Graph::Graph() {
  this->numVertices = 0;
  this->edges = new vector<Int64Pair>();
}

Matrix<REAL>* Graph::adjacencyMatrix(World* world, bool sparse) {
  auto attr = 0;
  if (sparse) {
    attr = SP;
  }
  auto A = new Matrix<REAL>(numVertices, numVertices,
      attr, *world, MAX_TIMES_SR);
  int64_t n = edges->size();
  int64_t idx[2 * n];
  REAL fill[2 * n];
  int64_t loc_n = 0;
  for (int64_t i = 0; i < n; ++i) {
    if (i % world->np == world->rank) {
      auto edge = (*edges)[i];
      idx[loc_n]      = edge.i2 * A->nrow + edge.i1;
      idx[loc_n + 1]  = edge.i1 * A->nrow + edge.i2;
      fill[loc_n]     = 1;
      fill[loc_n + 1] = 1;

      loc_n += 2;
    }
  }
  A->write(loc_n, idx, fill);
  return A;
}

Matrix <REAL> * read_matrix(World  &     dw,
                         int          n,
                         const char * fpath,
                         bool         remove_singlets,
                         int *        n_nnz,
                         int64_t      max_eREAL){
  uint64_t *my_edges = NULL;
  uint64_t my_nedges = 0;
  Semiring<REAL> s(MAX_REAL,
                  [](REAL a, REAL b){ return std::min(a,b); },
                  MPI_MIN,
                  0,
                  [](REAL a, REAL b){ return a+b; });
  if (dw.rank == 0) printf("Running MPI-IO graph reader n = %d... ",n);
  bool e_weights;
  // REAL *vals;
  std::vector<std::pair<uint64_t, uint64_t> > edges;
  std::vector<REAL> eweights;
  my_nedges = read_metis(dw.rank, dw.np, fpath, edges, &n, &e_weights, eweights);
  if (dw.rank == 0) printf("finished reading (%ld edges).\n", my_nedges);
  int64_t * inds = (int64_t*)malloc(sizeof(int64_t)*my_nedges);

  srand(dw.rank+1);
  for (int64_t i = 0; i < my_nedges; ++i){
    // printf("edge: %lld %lld %d\n", edges[i].first, edges[i].second, eweights[i]);
    inds[i] = edges[i].first + edges[i].second * n;
    if (!e_weights) {
      eweights.push_back(1);
      //vals[i] = 1;
      //vals[i] = (rand()%max_eREAL) + 1;
      //vals[i] = (rand()%10000) + 1;
    }
  }
  if (dw.rank == 0) printf("filling CTF graph\n");
  Matrix<REAL> * A_pre = new Matrix<REAL>(n, n, SP, dw, MAX_TIMES_SR, "A_rmat"); // MIN_TIMES_SR for PTAP
  A_pre->write(my_nedges, inds, eweights.data());
  (*A_pre)["ij"] += (*A_pre)["ji"];
  free(inds);
  // A_pre.print_matrix();
  // free(vals);
  
  return A_pre;
  //Matrix<REAL> newA =  preprocess_graph(n,dw,A_pre,remove_singlets,n_nnz,max_eREAL);
  //int64_t nprs;
  //newA.read_local_nnz(&nprs,&inds,&vals);

  //for (int64_t i=0; i<nprs; i++){
  //  printf("%d %d\n",inds[i]/newA.nrow,inds[i]%newA.nrow);
  //}
  //return newA;
}

Matrix <REAL> gen_uniform_matrix(World & dw,
                                int64_t n,
                                double  sp,
                                int64_t  max_eREAL){
  Semiring<REAL> s(MAX_REAL,
                  [](REAL a, REAL b){ return std::min(a,b); },
                  MPI_MIN,
                  0,
                  [](REAL a, REAL b){ return a+b; });

  //random adjacency matrix
  Matrix<REAL> A(n, n, SP, dw, s, "A");

  //fill with values in the range of [1,min(n*n,100)]
  srand(dw.rank+1);
//  A.fill_random(1, std::min(n*n,100));
  int nmy = ((int)std::max((int)(n*sp),(int)1))*((int)((n+dw.np-1)/dw.np));
  int64_t inds[nmy];
  REAL vals[nmy];
  int i=0;
  for (int64_t row=dw.rank*n/dw.np; row<(int)(dw.rank+1)*n/dw.np; row++){
    int64_t cols[std::max((int)(n*sp),1)];
    for (int64_t col=0; col<std::max((int)(n*sp),1); col++){
      bool is_rep;
      do {
        cols[col] = rand()%n;
        is_rep = 0;
        for (int c=0; c<col; c++){
          if (cols[c] == cols[col]) is_rep = 1;
        }
      } while (is_rep);
      inds[i] = cols[col]*n+row;
      vals[i] = (rand()%max_eREAL)+1;
      i++;
    }
  }
  A.write(i,inds,vals);

  A["ii"] = 0;

  //keep only values smaller than 20 (about 20% sparsity)
  //A.sparsify([=](int a){ return a<sp*100; });
   return A;
}


Matrix<REAL>* generate_kronecker(World* w, int order)
{
  auto g = new Graph();
  g->numVertices = 3;
  g->edges->emplace_back(0, 0);
  g->edges->emplace_back(0, 1);
  g->edges->emplace_back(1, 1);
  g->edges->emplace_back(1, 2);
  g->edges->emplace_back(2, 2);
  auto kinitiator = g->adjacencyMatrix(w);
  auto B = g->adjacencyMatrix(w);

  int64_t len = 1;
  int64_t matSize = 3;
  for (int i = 2; i <= order; i++) {
    len *= 3;
    int64_t lens[] = {3, len, 3, len};
    /**
    int * lens = new int[4];
    lens[0] = 3;
    lens[1] = len;
    lens[2] = 3;
    lens[3] = len;
    **/
    auto D = Tensor<REAL>(4, B->is_sparse, lens);
    D["ijkl"] = (*kinitiator)["ik"] * (*B)["jl"];

    matSize *= 3;
    auto B2 = new Matrix<REAL>(matSize, matSize, B->is_sparse * SP, *w, *B->sr);
    delete B;
    B2->reshape(D);
    B = B2;
    // B->print_matrix();
    // hook on B
  }
  (*B)["ii"] = 0;
  delete kinitiator;
  return B;
}
