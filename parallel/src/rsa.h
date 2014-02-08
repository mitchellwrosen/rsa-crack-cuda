#ifndef RSA_H_
#define RSA_H_

#include <gmp.h>

int rsa_compute_d(mpz_t d, const mpz_t n, const mpz_t p, const mpz_t q);
void rsa_phi(mpz_t phi_n, const mpz_t p, const mpz_t q);


#endif  // RSA_H_
