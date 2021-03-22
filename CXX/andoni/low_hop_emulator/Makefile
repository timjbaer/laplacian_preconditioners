include config.mk

all: tests debug

tests: test_low_hop_emulator test_subemulator test_ball

debug: debug_low_hop_emulator debug_subemulator debug_ball

ball.o: ball.h ball.cxx $(CTFDIR)
	$(CXX) $(CXXFLAGS) -c ball.cxx  $(DEFS) $(INCLUDES)

subemulator.o: subemulator.h subemulator.cxx ball.h $(CTFDIR)
	$(CXX) $(CXXFLAGS) -c subemulator.cxx $(DEFS) $(INCLUDES)

low_hop_emulator.o: low_hop_emulator.h low_hop_emulator.cxx subemulator.h ball.h $(CTFDIR)
	$(CXX) $(CXXFLAGS) -c low_hop_emulator.cxx $(DEFS) $(INCLUDES)

graph.o: ../../graph.h ../../graph.cxx ../../graph_io.h
	$(CXX) $(CXXFLAGS) -c ../../graph.cxx $(DEFS) $(INCLUDES)

test.o: ./test/test.h ./test/test.cxx ../../graph.h ../../graph_io.h $(CTFDIR)
	$(CXX) $(CXXFLAGS) -c ./test/test.cxx $(DEFS) $(INCLUDES)

test_ball: ./test/test_ball.cxx ball.o graph.o test.o $(CTFDIR)
	$(CXX) $(CXXFLAGS) -o test_ball ./test/test_ball.cxx ball.o graph.o test.o $(INCLUDES) $(LIB_PATH) $(LIBS)

debug_ball: ./test/test_ball.cxx ball.o graph.o test.o $(CTFDIR)
	$(CXX) $(CXXFLAGS) -DDEBUG -o debug_ball ./test/test_ball.cxx ball.o graph.o test.o $(INCLUDES) $(LIB_PATH) $(LIBS)

test_subemulator: ./test/test_subemulator.cxx subemulator.o ball.o graph.o test.o $(CTFDIR)
	$(CXX) $(CXXFLAGS) -o test_subemulator ./test/test_subemulator.cxx subemulator.o ball.o graph.o test.o $(INCLUDES) $(LIB_PATH) $(LIBS)

debug_subemulator: ./test/test_subemulator.cxx subemulator.o ball.o graph.o test.o $(CTFDIR)
	$(CXX) $(CXXFLAGS) -DDEBUG -o debug_subemulator ./test/test_subemulator.cxx subemulator.o ball.o graph.o test.o $(INCLUDES) $(LIB_PATH) $(LIBS)

test_low_hop_emulator: ./test/test_low_hop_emulator.cxx low_hop_emulator.o subemulator.o ball.o graph.o test.o $(CTFDIR)
	$(CXX) $(CXXFLAGS) -o test_low_hop_emulator ./test/test_low_hop_emulator.cxx low_hop_emulator.o subemulator.o ball.o graph.o test.o $(INCLUDES) $(LIB_PATH) $(LIBS)

debug_low_hop_emulator: ./test/test_low_hop_emulator.cxx low_hop_emulator.o subemulator.o ball.o graph.o test.o $(CTFDIR)
	$(CXX) $(CXXFLAGS) -DDEBUG -o debug_low_hop_emulator ./test/test_low_hop_emulator.cxx low_hop_emulator.o subemulator.o ball.o graph.o test.o $(INCLUDES) $(LIB_PATH) $(LIBS)

clean:
	rm -f *.o test_low_hop_emulator debug_low_hop_emulator test_subemulator debug_subemulator test_ball debug_ball
