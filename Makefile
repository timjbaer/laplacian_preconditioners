include config.mk

all: test

test.o: alg_graph.o mst.o test.h test.cxx $(CTFDIR)
	$(CXX) $(CXXFLAGS) -c test.cxx mst.o alg_graph.o $(INCLUDES)

test_connectivity: alg_graph.o graph_io.o graph_gen.o connectivity.o mst.o test.o test_connectivity.cxx $(CTFDIR)
	$(CXX) $(CXXFLAGS) -o test_connectivity test_connectivity.cxx test.o connectivity.o mst.o graph_io.o graph_gen.o alg_graph.o $(INCLUDES) $(LIB_PATH) $(LIBS)

graph.o: graph.h $(CTFDIR)
	$(CXX) $(CXXFLAGS) -c graph.cxx $(INCLUDES)

multigrid.o: graph.o multigrid.h
	$(CXX) $(CXXFLAGS) -c multigrid.cxx $(INCLUDES)

test: graph.o multigrid.o test.cxx
	$(CXX) $(CXXFLAGS) -o test test.cxx multigrid.o graph.o $(INCLUDES) $(LIB_PATH) $(LIBS)

clean:
	rm -rf graph.o multigrid.o test
