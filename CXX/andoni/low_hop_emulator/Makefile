include config.mk

all: test

ball.o: ball.h ball.cxx $(CTFDIR)
	$(CXX) $(CXXFLAGS) -c ball.cxx  $(DEFS) $(INCLUDES)

bourgain.o: bourgain.h bourgain.cxx ball.h $(CTFDIR)
	$(CXX) $(CXXFLAGS) -c bourgain.cxx $(DEFS) $(INCLUDES)

low_hop_emulator.o: low_hop_emulator.h low_hop_emulator.cxx ball.h bourgain.h $(CTFDIR)
	$(CXX) $(CXXFLAGS) -c low_hop_emulator.cxx $(DEFS) $(INCLUDES)

graph.o: ../../graph.h ../../graph.cxx ../../graph_io.h
	$(CXX) $(CXXFLAGS) -c ../../graph.cxx $(DEFS) $(INCLUDES)

test: test.cxx low_hop_emulator.o bourgain.o ball.o graph.o $(CTFDIR)
	$(CXX) $(CXXFLAGS) -o test test.cxx low_hop_emulator.o bourgain.o ball.o graph.o $(INCLUDES) $(LIB_PATH) $(LIBS)

clean:
	rm -f *.o test
