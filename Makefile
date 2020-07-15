include config.mk

all: test

graph_io.o: graph_io.h $(CTFDIR)
	$(CXX) $(CXXFLAGS) -c graph_io.cxx $(INCLUDES)

graph.o: graph_io.o graph.h $(CTFDIR)
	$(CXX) $(CXXFLAGS) -c graph.cxx $(INCLUDES)

multigrid.o: graph_io.o graph.o multigrid.h
	$(CXX) $(CXXFLAGS) -c multigrid.cxx $(INCLUDES)

test: graph_io.o graph.o multigrid.o test.cxx
	$(CXX) $(CXXFLAGS) -o test test.cxx multigrid.o graph.o graph_io.o $(INCLUDES) $(LIB_PATH) $(LIBS)

clean:
	rm -rf graph_io.o graph.o multigrid.o test
