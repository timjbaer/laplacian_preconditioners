include config.mk

all: test

graph.o: graph.h $(CTFDIR)
	$(CXX) $(CXXFLAGS) -c graph.cxx $(INCLUDES)

multigrid.o: graph.o multigrid.h
	$(CXX) $(CXXFLAGS) -c multigrid.cxx $(INCLUDES)

test: graph.o multigrid.o test.cxx
	$(CXX) $(CXXFLAGS) -o test test.cxx multigrid.o graph.o $(INCLUDES) $(LIB_PATH) $(LIBS)

clean:
	rm -rf graph.o multigrid.o test
