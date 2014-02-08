#include "rsa.h"

#include <stdlib.h>
#include <gmp.h>

// Compute private key d, given n, p, and q. Assume e = 2^16+1
int rsa_compute_d(mpz_t d, const mpz_t n, const mpz_t p, const mpz_t q) {
  mpz_t e, phi_n, gcd;
  mpz_inits(e, phi_n, gcd, '\0');

  rsa_phi(phi_n, p, q);
  mpz_set_ui(e, 65537);

  if (mpz_invert(d, e, phi_n) == 0)
    exit(EXIT_FAILURE);

  mpz_clears(phi_n, gcd, '\0');
  return 0;
}

// Calculate euler's totient of n, assuming n = p*q. Then,
// phi(n) = phi(p)phi(q) = (p-1)(q-1)
void rsa_phi(mpz_t phi_n, const mpz_t p, const mpz_t q) {
  mpz_t p_minus_one, q_minus_one;
  mpz_inits(p_minus_one, q_minus_one, '\0');

  mpz_sub_ui(p_minus_one, p, 1);
  mpz_sub_ui(q_minus_one, q, 1);
  mpz_mul(phi_n, p_minus_one, q_minus_one);

  mpz_clears(p_minus_one, q_minus_one, '\0');
}
