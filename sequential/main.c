#include <stdio.h>
#include <stdlib.h>

#include "mygmp.h"

void printUsage(char* progname);

int main(int argc, char** argv) {
  if (argc != 3)
    printUsage(argv[0]);

  int numKeys = atoi(argv[2]);
  mpz_t* keys = mpz_reads(argv[1], numKeys);

  mpz_t gcd;
  mpz_init(gcd);
  for (int i = 0; i < numKeys-1; ++i) {
    fprintf(stdout, "%d/%d\n", i, numKeys-2);
    for (int j = i+1; j < numKeys; ++j) {
      mpz_gcd(gcd, keys[i], keys[j]);
      if (mpz_cmp_ui(gcd, 1) != 0)
        gmp_fprintf(stdout, "%d,%d: %Zd\n", i, j, gcd);
    }
  }
}

void printUsage(char* progname) {
  fprintf(stderr, "Usage: %s keyfile numkeys\n", progname);
  exit(EXIT_FAILURE);
}
