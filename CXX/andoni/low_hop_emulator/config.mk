CTFDIR    = /Users/timbaer/lpna/raghavendrak-ctf
MPI_DIR   =
CXX       = mpicxx -cxx=g++
OPTS      = -O0 -g
#CXXFLAGS  = -std=c++0x -fopenmp $(OPTS) -Wall -DPROFILE -DPMPI -DMPIIO
CXXFLAGS  = -std=c++0x $(OPTS) -Wall -Wno-format -DPMPI -DMPIIO
INCLUDES  = -I$(CTFDIR)/include
#INCLUDES  = -I$(CTFDIR)/include -I/Users/timbaer/lpna/critter/include
LIBS      = -L$(CTFDIR)/lib -lctf -lblas ../../generator/libgraph_generator_mpi.a -llapack -lblas
#LIBS      = -L$(CTFDIR)/lib -lctf -lblas ../../generator/libgraph_generator_mpi.a -llapack -lblas -L/Users/timbaer/lpna/critter/lib/libcritter.a
#LIBS      = -lctf -lblas generator/libgraph_generator_mpi.a -llapack -lblas 
DEFS      =
CUDA_ARCH = sm_37
NVCC      = $(CXX)
NVCCFLAGS = $(CXXFLAGS)
