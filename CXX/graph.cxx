#include "graph.h"
#include "generator/make_graph.h"

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

Graph::~Graph() {
  delete this->edges;
}

Matrix<REAL>* Graph::adjacencyMatrix(World* world, bool sparse) {
  auto attr = 0;
  if (sparse) {
    attr = SP;
  }
  auto A = new Matrix<REAL>(numVertices, numVertices,
      attr, *world, PLUS_TIMES_SR);
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

uint64_t gen_graph(int scale, int edgef, uint64_t seed, uint64_t **edges) {
  uint64_t nedges;
  double   initiator[4] = {.57, .19, .19, .05};
  make_graph(scale, (((int64_t)1)<<scale)*edgef, seed, seed+1, initiator, (int64_t *)&nedges, (int64_t **)edges);

  return nedges;
}

Matrix <REAL> * preprocess_graph(int         n,
                               World &        dw,
                               Matrix<REAL> * A_pre,
                               bool           remove_singlets,
                               int *          n_nnz,
                               int64_t        max_ewht){
  Semiring<REAL> s(MAX_REAL,
                  [](REAL a, REAL b){ return std::min(a,b); },
                  MPI_MIN,
                  0,
                  [](REAL a, REAL b){ return a+b; });

  (*A_pre)["ii"] = 0;

  A_pre->sparsify([](int a){ return a>0; });

  if (dw.rank == 0)
    printf("A contains %ld nonzeros\n", A_pre->nnz_tot);

  if (remove_singlets){
    Vector<int> rc(n, dw);
    rc["i"] += ((Function<REAL>)([](REAL a){ return (int)(a>0); }))((*A_pre)["ij"]);
    rc["i"] += ((Function<REAL>)([](REAL a){ return (int)(a>0); }))((*A_pre)["ji"]);
    int * all_rc; // = (int*)malloc(sizeof(int)*n);
    int64_t nval;
    rc.read_all(&nval, &all_rc);
    int n_nnz_rc = 0;
    int n_single = 0;
    for (int i=0; i<nval; i++){
      if (all_rc[i] != 0){
        if (all_rc[i] == 2) n_single++;
        all_rc[i] = n_nnz_rc;
        n_nnz_rc++;
      } else {
        all_rc[i] = -1;
      }
    }
    if (dw.rank == 0) printf("n_nnz_rc = %d of %d vertices kept, %d are 0-degree, %d are 1-degree\n", n_nnz_rc, n,(n-n_nnz_rc),n_single);
    Matrix<REAL> * A = new Matrix<REAL>(n_nnz_rc, n_nnz_rc, SP, dw, PLUS_TIMES_SR, "A"); // MIN_TIMES_SR for PTAP
    int * pntrs[] = {all_rc, all_rc};

    A->permute(0, *A_pre, pntrs, 1);
    delete A_pre;
    free(all_rc);
    if (dw.rank == 0) printf("preprocessed matrix has %ld edges\n", A->nnz_tot);

    (*A)["ii"] = 0;
    *n_nnz = n_nnz_rc;
    return A;
  } else {
    *n_nnz= n;
    (*A_pre)["ii"] = 0;
    //A_pre.print();
    return A_pre;
  }
//  return n_nnz_rc;

}

Matrix<REAL> * read_matrix(World  &     dw,
                           int          n,
                           const char * fpath,
                           bool         remove_singlets,
                           int *        n_nnz,
                           int64_t      max_ewht){
  uint64_t *my_edges = NULL;
  uint64_t my_nedges = 0;
  if (dw.rank == 0) printf("Running MPI-IO graph reader n = %d... ",n);
  bool e_weights;
  // REAL *vals;
  std::vector<std::pair<uint64_t, uint64_t> > edges;
  std::vector<REAL> eweights;
  // my_nedges = read_metis(dw.rank, dw.np, fpath, edges, &n, &e_weights, eweights); // TODO: port over graph_io.cxx from CTF_connectivity
  if (dw.rank == 0) printf("finished reading (%ld edges).\n", my_nedges);
  int64_t * inds = (int64_t*)malloc(sizeof(int64_t)*my_nedges);

  srand(dw.rank+1);
  for (int64_t i = 0; i < my_nedges; ++i){
    // printf("edge: %lld %lld %d\n", edges[i].first, edges[i].second, eweights[i]);
    inds[i] = edges[i].first + edges[i].second * n;
    if (!e_weights) {
      eweights.push_back(1);
      //vals[i] = 1;
      //vals[i] = (rand()%max_ewht) + 1;
      //vals[i] = (rand()%10000) + 1;
    }
  }
  if (dw.rank == 0) printf("filling CTF graph\n");
  Matrix<REAL> * A_pre = new Matrix<REAL>(n, n, SP, dw, PLUS_TIMES_SR, "A_rmat"); // MIN_TIMES_SR for PTAP
  A_pre->write(my_nedges, inds, eweights.data());
  (*A_pre)["ij"] += (*A_pre)["ji"];
  free(inds);
  // A_pre.print_matrix();
  // free(vals);

  Matrix<REAL> * newA = preprocess_graph(n,dw,A_pre,remove_singlets,n_nnz,max_ewht);
  delete A_pre;
  //int64_t nprs;
  //newA.read_local_nnz(&nprs,&inds,&vals);

  //for (int64_t i=0; i<nprs; i++){
  //  printf("%d %d\n",inds[i]/newA.nrow,inds[i]%newA.nrow);
  //}
  return newA;
}

Matrix <REAL> * gen_rmat_matrix(World  & dw,
                                int      scale,
                                int      ef,
                                uint64_t gseed,
                                bool     remove_singlets,
                                int *    n_nnz,
                                int64_t  max_ewht){
  uint64_t *edge=NULL;
  uint64_t nedges = 0;
  //random adjacency matrix
  int n = pow(2,scale);
  Matrix<REAL> * A_pre = new Matrix<REAL>(n, n, SP, dw, PLUS_TIMES_SR, "A_rmat");
  if (dw.rank == 0) printf("Running graph generator n = %d... ",n);
  nedges = gen_graph(scale, ef, gseed, &edge);
  if (dw.rank == 0) printf("done.\n");
  int64_t * inds = (int64_t*)malloc(sizeof(int64_t)*nedges);
  REAL * vals = (REAL*)malloc(sizeof(REAL)*nedges);

  srand(dw.rank+1);
  for (int64_t i=0; i<nedges; i++){
    inds[i] = (edge[2*i]+(edge[2*i+1])*n);
    //vals[i] = (rand()%max_ewht) + 1;
    //vals[i] = 1;
    vals[i] = (rand()%100) + 1;
  }
  if (dw.rank == 0) printf("filling CTF graph\n");
  A_pre->write(nedges,inds,vals);
  (*A_pre)["ij"] += (*A_pre)["ji"];
  free(inds);
  free(vals);
  free(edge);

  return preprocess_graph(n,dw,A_pre,remove_singlets,n_nnz,max_ewht);
}

Matrix <REAL> gen_uniform_matrix(World & dw,
                                int64_t n,
                                double  sp,
                                int64_t  max_ewht){
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
      vals[i] = (rand()%max_ewht)+1;
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
  Matrix<REAL> * kinitiator = g->adjacencyMatrix(w);
  Matrix<REAL> * B = g->adjacencyMatrix(w);

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
  delete kinitiator;
  delete g;
  return B;
}
