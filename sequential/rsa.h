#ifndef RSA_H_
#define RSA_H_

#include "mygmp.h"

// Compute modulus n, encryption exponent e, and decryption exponent d.
// Returns 0 on success, -1 on failure.
int rsa_keygen512(mpz_t n, mpz_t e, mpz_t d);

// Compute encryption/decryption keys e and d, given the prime factors
// p and q of n.
int rsa_compute_keys(mpz_t e, mpz_t d, const mpz_t n, const mpz_t p, const mpz_t q);

// c = m^e mod n
void rsa_encrypt(mpz_t c, const mpz_t m, const mpz_t e, const mpz_t n);

// m = c^d mod n
void rsa_decrypt(mpz_t m, const mpz_t c, const mpz_t d, const mpz_t n);

#endif  // RSA_H_
