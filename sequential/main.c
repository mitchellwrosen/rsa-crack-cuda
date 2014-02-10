#include <stdio.h>
#include <stdlib.h>

#include "rsa.h"
#include "mygmp.h"

static int *cracked = NULL;
static int crackedLen = 0;

void printUsage(char* progname);

int checkIfCrackedAlready(int n) {
  for (int i = 0; i < crackedLen; ++i) {
    if (n == cracked[i])
      return 1;
  }

  return 0;
}

int main(int argc, char** argv) {
  if (argc < 3)
    printUsage(argv[0]);

  int numKeys = atoi(argv[2]);
  mpz_t* keys = mpz_reads(argv[1], numKeys);

  FILE *stream = argc == 4 ? fopen(argv[3], "w") : stdout;

  mpz_t gcd, q1, q2, d1, d2;
  mpz_inits(gcd, q1, q2, d1, d2, NULL);

  cracked = malloc(numKeys * sizeof(int));

  for (int i = 0; i < numKeys-1; ++i) {
    for (int j = i+1; j < numKeys; ++j) {
      mpz_gcd(gcd, keys[i], keys[j]);
      if (mpz_cmp_ui(gcd, 1) != 0) {
        int crackedN1 = checkIfCrackedAlready(i);
        int crackedN2 = checkIfCrackedAlready(j);

        if (!crackedN1 || !crackedN2) {
          if (!crackedN1) {
            mpz_divexact(q1, keys[i], gcd);
            rsa_compute_d(d1, keys[i], gcd, q1);
            mpz_out_str(stream, 10, keys[i]);
            fputc(':', stream);
            mpz_out_str(stream, 10, d1);
            fputc('\n', stream);

            cracked[crackedLen++] = i;
          }

          if (!crackedN2) {
            mpz_divexact(q2, keys[j], gcd);
            rsa_compute_d(d2, keys[j], gcd, q2);
            mpz_out_str(stream, 10, keys[j]);
            fputc(':', stream);
            mpz_out_str(stream, 10, d2);
            fputc('\n', stream);

            cracked[crackedLen++] = j;
          }
        }
      }
    }
  }

  free(keys);
  free(cracked);

  if (argc == 4)
    fclose(stream);

  return 0;
}

void printUsage(char* progname) {
  fprintf(stderr, "Usage: %s keyfile numkeys [outputfile]\n", progname);
  exit(EXIT_FAILURE);
}
