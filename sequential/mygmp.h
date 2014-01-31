#ifndef GMP_H_
#define GMP_H_

#include <stdio.h>
#include <gmp.h>

typedef enum
  { COMPOSITE  = 0
  , PROB_PRIME = 1
  , PRIME      = 2
  } Primality;

// 5 Integer Functions

// 5.9 Number Theoretic Functions
Primality mpz_primality(mpz_t n);

// 5.12 Input and Output Functions
size_t mpz_read(const char* path, mpz_t n);
size_t mpz_write(const char* path, const mpz_t n);

// Read keys from a file, decimal format, one per line.
mpz_t* mpz_reads(const char* path, int n);

// Read/write a single key from a stream, in raw format.
size_t mpz_fread(FILE* stream, mpz_t n);
size_t mpz_fwrite(FILE* stream, const mpz_t n);

size_t mpz_print(const mpz_t n);
size_t mpz_printb(const mpz_t n, int base);

// 9 Random Number Functions
// 9.1 Random State Initialization
// 9.2 Random State Seeding
// Combine initialization and seed.
void gmp_rand_init_seed(gmp_randstate_t rstate, unsigned long int seed);

// Generate a prime number.
void gmp_genprime(mpz_t n, gmp_randstate_t rstate, mp_bitcnt_t bitcount);

#endif  // GMP_H_
