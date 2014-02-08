UNAME := $(shell uname)

NVCC=nvcc
GPP=g++
SOURCEDIR=src/

NVFLAGS=-g -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35

ifeq ($(UNAME), Darwin)
LDFLAGS=-L. -L/usr/local/cuda/lib -lcudart -lgmp -I.
GPPFLAGS=-g -Wno-write-strings -I/usr/local/cuda/include
else
LDFLAGS=-L. -L/usr/local/cuda/lib64 -L/home/clupo/gmp/lib -lcudart -lgmp -I.
GPPFLAGS=-O3 -g -Wno-write-strings -I/home/clupo/gmp/include -I/usr/local/cuda/include
endif

all: main

main: main.o rsa.o integer.o
	$(GPP) $(LDFLAGS) $^ -o $@

main.o: $(SOURCEDIR)main.cpp
	$(GPP) $(GPPFLAGS) -c $< -o $@

rsa.o: $(SOURCEDIR)rsa.cpp
	$(GPP) $(GPPFLAGS) -c $< -o $@

integer.o: $(SOURCEDIR)integer.cu $(SOURCEDIR)integer.h
	$(NVCC) $(NVFLAGS) -c $< -o $@

clean:
	rm -f *.o main
