CC=gcc
CFLAGS=-Wall -std=c99 -g
LDFLAGS=-Wall

INC=-I.
LIBS=-lgmp

all: main

main: main.o mygmp.o
	$(CC) $(LDFLAGS) $(INC) $(LIBS) $^ -o $@

%.o: %.c
	$(CC) $(INC) $(CFLAGS) -c $< -o $@

clean:
	@rm -f *.o *.gch main
