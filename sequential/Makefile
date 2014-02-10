CC=gcc
CFLAGS=-Wall -std=c99 -O3
LDFLAGS=-Wall -L/home/clupo/gmp/lib

INC=-I. -I/home/clupo/gmp/include
LIBS=-lgmp

all: main

main: main.o mygmp.o rsa.o
	$(CC) $(LDFLAGS) $(INC) $(LIBS) $^ -o $@

%.o: %.c
	$(CC) $(INC) $(CFLAGS) -c $< -o $@

clean:
	@rm -f *.o *.gch main
