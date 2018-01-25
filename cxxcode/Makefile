CC=g++
#CC=g++
#CC=clang++
LIBS=-L/scratch/cluster/elisaf/dynet/build/dynet -ldynet -lm -lboost_serialization -lboost_filesystem -lboost_system -lboost_random -lboost_program_options
CFLAGS=-I/scratch/cluster/elisaf/dynet -I/scratch/cluster/elisaf/eigen -I/scratch/cluster/elisaf/easyloggingpp-9.95.3/src -std=gnu++11 -Wall
OBJ=main.o util.o

all: dtc

%.o: %.cc
	$(CC) -v $(CFLAGS) -c -o $@ $< -L/scratch/cluster/elisaf/dynet/build/dynet

dtc: main.o util.o
	$(CC) -v  /scratch/cluster/elisaf/easyloggingpp-9.95.3/src/easylogging++.cc -std=gnu++11 $(LIBS) $^ -o $@ -L/scratch/cluster/elisaf/dynet/build/dynet -ldynet -lm -lboost_serialization -lboost_filesystem -lboost_system -lboost_random -lboost_program_options

clean:
	rm -rf *.o *.*~ dtc
