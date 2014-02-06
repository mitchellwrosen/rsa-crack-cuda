UNAME := $(shell uname)

NVCC=nvcc
GPP=g++
SOURCEDIR=src/


ifeq ($(UNAME), Darwin)
NVFLAGS=-O3 -g -arch=sm_30
LDFLAGS=-L. -L/usr/local/cuda/lib -lcudart -lgmp
GPPFLAGS=-O3 -g
else
NVFLAGS=-O3 -g -arch=sm_20
LDFLAGS=-L. -L/usr/local/cuda/lib64 -L/home/clupo/gmp/lib -lcudart -lgmp
GPPFLAGS=-O3 -g -I/home/clupo/gmp/include
endif

all: main

main: main.o integer.o
	$(GPP) $(LDFLAGS) $^ -o $@

main.o: $(SOURCEDIR)main.cpp
	$(GPP) $(GPPFLAGS) -c $< -o $@

integer.o: $(SOURCEDIR)integer.cu $(SOURCEDIR)integer.h
	$(NVCC) $(NVFLAGS) -c $< -o $@

clean:
	rm -f *.o main
