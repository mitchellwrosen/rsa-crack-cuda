CC=nvcc
CFLAGS=-O3 -g -rdc=true -gencode arch=compute_30,code=sm_30
LDFLAGS=-arch=sm_30

all: main

main: main.o kernel.o integer.o
	$(CC) $(LDFLAGS) $^ -o $@

%.o: %.cu
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o main
