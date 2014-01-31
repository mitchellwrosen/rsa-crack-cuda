#include "mygmp.h"

#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>

// 5 Integer Functions
// 5.1 Initialization Functions

// 5.9 Number Theoretic Functions
Primality mpz_primality(mpz_t n) {
  return mpz_probab_prime_p(n, 25);
}

// 5.12 Input and Output Functions
size_t mpz_read(const char* path, mpz_t n) {
  FILE* file = fopen(path, "r");
  if (file == NULL)
    return -1;

  size_t nbytes = mpz_fread(file, n);
  if ((fclose(file) != 0))
    return -1;

  return nbytes;
}

size_t mpz_write(const char* path, const mpz_t n) {
  FILE* file = fopen(path, "w");
  if (file == NULL)
    return -1;

  size_t nbytes = mpz_fwrite(file, n);
  if ((fclose(file) != 0))
      return -1;

  return nbytes;
}

mpz_t* mpz_reads(const char* path, int n) {
  FILE* file = fopen(path, "r");
  if (file == NULL)
    return NULL;

  mpz_t* ints = (mpz_t*) calloc(n, sizeof(mpz_t));
  for (int i = 0; i < n; ++i)
    gmp_fscanf(file, "%Zd\n", ints[i]);

  fclose(file);
  return ints;
}

size_t mpz_fread(FILE* stream, mpz_t n) {
  return mpz_inp_raw(n, stream);
}

size_t mpz_fwrite(FILE* stream, const mpz_t n) {
  return mpz_out_raw(stream, n);
}

size_t mpz_print(const mpz_t n) {
  return mpz_printb(n, 10);
}

size_t mpz_printb(const mpz_t n, int base) {
  size_t nbytes = mpz_out_str(stdout, base, n);
  fprintf(stdout, "\n");
  return nbytes+1;
}

// 9 Random Number Functions
// 9.1 Random State Initialization
// 9.2 Random State Seeding
void gmp_rand_init_seed(gmp_randstate_t rstate, unsigned long int seed) {
  gmp_randinit_default(rstate);
  gmp_randseed_ui(rstate, seed);
}

void gmp_genprime(mpz_t n, gmp_randstate_t rstate, mp_bitcnt_t bitcount) {
  do {
    mpz_urandomb(n, rstate, bitcount);
  } while (mpz_primality(n) != PRIME);
}
