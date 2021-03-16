include config.mk

all: test_low_hop_em debug_low_hop_em test_ball debug_ball

ball.o: ball.h ball.cxx $(CTFDIR)
	$(CXX) $(CXXFLAGS) -c ball.cxx  $(DEFS) $(INCLUDES)

bourgain.o: bourgain.h bourgain.cxx ball.h $(CTFDIR)
	$(CXX) $(CXXFLAGS) -c bourgain.cxx $(DEFS) $(INCLUDES)

low_hop_emulator.o: low_hop_emulator.h low_hop_emulator.cxx ball.h $(CTFDIR)
	$(CXX) $(CXXFLAGS) -c low_hop_emulator.cxx $(DEFS) $(INCLUDES)

graph.o: ../../graph.h ../../graph.cxx ../../graph_io.h
	$(CXX) $(CXXFLAGS) -c ../../graph.cxx $(DEFS) $(INCLUDES)

test.o: test.h test.cxx ../../graph.h $(CTFDIR)
	$(CXX) $(CXXFLAGS) -c test.cxx $(DEFS) $(INCLUDES)

test_ball: test_ball.cxx ball.o graph.o test.o $(CTFDIR)
	$(CXX) $(CXXFLAGS) -o test_ball test_ball.cxx ball.o graph.o test.o $(INCLUDES) $(LIB_PATH) $(LIBS)

debug_ball: test_ball.cxx ball.o graph.o test.o $(CTFDIR)
	$(CXX) $(CXXFLAGS) -DDEBUG -o debug_ball test_ball.cxx ball.o graph.o test.o $(INCLUDES) $(LIB_PATH) $(LIBS)

test_low_hop_em: test_low_hop_em.cxx low_hop_emulator.o bourgain.o ball.o graph.o test.o $(CTFDIR)
	$(CXX) $(CXXFLAGS) -o test_low_hop_em test_low_hop_em.cxx low_hop_emulator.o bourgain.o ball.o graph.o test.o $(INCLUDES) $(LIB_PATH) $(LIBS)

debug_low_hop_em: test_low_hop_em.cxx low_hop_emulator.o bourgain.o ball.o graph.o test.o $(CTFDIR)
	$(CXX) $(CXXFLAGS) -DDEBUG -o debug_low_hop_em test_low_hop_em.cxx low_hop_emulator.o bourgain.o ball.o graph.o test.o $(INCLUDES) $(LIB_PATH) $(LIBS)

clean:
	rm -f *.o test_low_hop_em debug_low_hop_em test_ball debug_ball
