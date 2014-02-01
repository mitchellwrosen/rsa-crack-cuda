CC=nvcc
CFLAGS=-O3 -g -rdc=true -arch=sm_21 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35
LDFLAGS=-arch=sm_21

all: main

main: main.o integer.o
	$(CC) $(LDFLAGS) $^ -o $@

%.o: %.cu
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o main
