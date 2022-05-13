include config.mk

all: tests debug

tests: test_low_hop_emulator test_subemulator test_ball unit_tests run_test_ball

debug: debug_low_hop_emulator debug_subemulator debug_ball

bench: bench_low_hop_emulator bench_subemulator bench_ball

shared.o: shared.h shared.cxx $(CTFDIR)
	$(CXX) $(CXXFLAGS) -c shared.cxx  $(DEFS) $(INCLUDES)

ball.o: ball.h ball.cxx shared.h $(CTFDIR)
	$(CXX) $(CXXFLAGS) -c ball.cxx  $(DEFS) $(INCLUDES)

subemulator.o: subemulator.h subemulator.cxx ball.h shared.h $(CTFDIR)
	$(CXX) $(CXXFLAGS) -c subemulator.cxx $(DEFS) $(INCLUDES)

low_hop_emulator.o: low_hop_emulator.h low_hop_emulator.cxx subemulator.h ball.h shared.h $(CTFDIR)
	$(CXX) $(CXXFLAGS) -c low_hop_emulator.cxx $(DEFS) $(INCLUDES)

graph.o: ../../graph.h ../../graph.cxx ../../graph_io.h
	$(CXX) $(CXXFLAGS) -c ../../graph.cxx $(DEFS) $(INCLUDES)

test.o: ./test/test.h ./test/test.cxx ../../graph.h ../../graph_io.h shared.h $(CTFDIR)
	$(CXX) $(CXXFLAGS) -c ./test/test.cxx $(DEFS) $(INCLUDES)

test_ball.o: ./test/test_ball.cxx ./test/test_ball.h shared.o ball.o graph.o test.o $(CTFDIR)
	$(CXX) $(CXXFLAGS) -c $< $(DEFS) $(INCLUDES)

test_ball: ./test/run_test_ball.cxx test_ball.o shared.o ball.o graph.o test.o $(CTFDIR)
	$(CXX) $(CXXFLAGS) -DTEST -o test_ball $< test_ball.o shared.o ball.o graph.o test.o $(INCLUDES) $(LIB_PATH) $(LIBS)

debug_ball: ./test/test_ball.cxx shared.o ball.o graph.o test.o $(CTFDIR)
	$(CXX) $(CXXFLAGS) -DDEBUG -o debug_ball ./test/test_ball.cxx shared.o ball.o graph.o test.o $(INCLUDES) $(LIB_PATH) $(LIBS)

bench_ball: ./test/test_ball.cxx shared.o ball.o graph.o test.o $(CTFDIR)
	$(CXX) $(CXXFLAGS) -o bench_ball ./test/test_ball.cxx shared.o ball.o graph.o test.o $(INCLUDES) $(LIB_PATH) $(LIBS)

test_subemulator: ./test/test_subemulator.cxx subemulator.o shared.o ball.o graph.o test.o $(CTFDIR)
	$(CXX) $(CXXFLAGS) -DTEST -o test_subemulator ./test/test_subemulator.cxx subemulator.o shared.o ball.o graph.o test.o $(INCLUDES) $(LIB_PATH) $(LIBS)

debug_subemulator: ./test/test_subemulator.cxx subemulator.o shared.o ball.o graph.o test.o $(CTFDIR)
	$(CXX) $(CXXFLAGS) -DDEBUG -o debug_subemulator ./test/test_subemulator.cxx subemulator.o shared.o ball.o graph.o test.o $(INCLUDES) $(LIB_PATH) $(LIBS)

bench_subemulator: ./test/test_subemulator.cxx subemulator.o shared.o ball.o graph.o test.o $(CTFDIR)
	$(CXX) $(CXXFLAGS) -o bench_subemulator ./test/test_subemulator.cxx subemulator.o shared.o ball.o graph.o test.o $(INCLUDES) $(LIB_PATH) $(LIBS)

test_low_hop_emulator: ./test/test_low_hop_emulator.cxx low_hop_emulator.o subemulator.o shared.o ball.o graph.o test.o $(CTFDIR)
	$(CXX) $(CXXFLAGS) -DTEST -o test_low_hop_emulator ./test/test_low_hop_emulator.cxx low_hop_emulator.o subemulator.o shared.o ball.o graph.o test.o $(INCLUDES) $(LIB_PATH) $(LIBS)

debug_low_hop_emulator: ./test/test_low_hop_emulator.cxx low_hop_emulator.o subemulator.o shared.o ball.o graph.o test.o $(CTFDIR)
	$(CXX) $(CXXFLAGS) -DDEBUG -o debug_low_hop_emulator ./test/test_low_hop_emulator.cxx low_hop_emulator.o subemulator.o shared.o ball.o graph.o test.o $(INCLUDES) $(LIB_PATH) $(LIBS)

bench_low_hop_emulator: ./test/test_low_hop_emulator.cxx low_hop_emulator.o subemulator.o shared.o ball.o graph.o test.o $(CTFDIR)
	$(CXX) $(CXXFLAGS) -o bench_low_hop_emulator ./test/test_low_hop_emulator.cxx low_hop_emulator.o subemulator.o shared.o ball.o graph.o test.o $(INCLUDES) $(LIB_PATH) $(LIBS)

unit_tests: ./test/unit_tests.cxx test_ball.o shared.o ball.o graph.o test.o $(CTFDIR)
	$(CXX) $(CXXFLAGS) -DTEST -o unit_tests $< test_ball.o shared.o ball.o graph.o test.o $(INCLUDES) $(LIB_PATH) $(LIBS)

clean:
	rm -f *.o test_low_hop_emulator debug_low_hop_emulator bench_low_hop_emulator test_subemulator debug_subemulator bench_subemulator test_ball debug_ball bench_ball unit_tests
