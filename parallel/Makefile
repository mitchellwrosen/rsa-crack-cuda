CC=nvcc
NVFLAGS=-O3 -g -rdc=true -gencode arch=compute_30,code=sm_30
LDFLAGS=-arch=sm_30

all: main

main: main.o kernel.o integer.o
	$(CC) $(LDFLAGS) $^ -o $@

%.o: %.cu
	$(CC) $(NVFLAGS) -c $< -o $@

clean:
	rm -f *.o main
