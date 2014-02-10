#include "rsa.h"

#include <time.h>
#include <stdlib.h>

#include "mygmp.h"

int rsa_keygen(mpz_t n, mpz_t e, mpz_t d, mp_bitcnt_t bitcount);
void rsa_generate_npq(mpz_t n, mpz_t p, mpz_t q, mp_bitcnt_t bitcount);
void rsa_phi(mpz_t phi_n, const mpz_t p, const mpz_t q);

int rsa_keygen512(mpz_t n, mpz_t e, mpz_t d) {
  return rsa_keygen(n, e, d, 512);
}

int rsa_keygen(mpz_t n, mpz_t e, mpz_t d, mp_bitcnt_t bitcount) {
  mpz_t p, q;
  mpz_inits(p, q, '\0');

  rsa_generate_npq(n, p, q, bitcount);
  int ret = rsa_compute_keys(e, d, n, p, q);

  mpz_clears(p, q, '\0');
  return ret;
}

void rsa_generate_npq(mpz_t n, mpz_t p, mpz_t q, mp_bitcnt_t bitcount) {
  // Initialize random generator.
  gmp_randstate_t rstate;
  gmp_rand_init_seed(rstate, time(NULL));

  // Generate p,q.
  gmp_genprime(p, rstate, bitcount);
  gmp_genprime(q, rstate, bitcount);

  // n = p*q
  mpz_mul(n, p, q);
}

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


int rsa_compute_keys(mpz_t e, mpz_t d, const mpz_t n, const mpz_t p, const mpz_t q) {
  mpz_t phi_n, gcd;
  mpz_inits(phi_n, gcd, '\0');

  rsa_phi(phi_n, p, q);

  // choose e s.t. 1 < e < phi_n; gcd(e,n) = 1
  mpz_set_ui(e, 65537); // Good starting value for e
  while (1) {
    if (mpz_cmp(e, phi_n) >= 0) {   // Have we exhausted all es?
      mpz_clears(phi_n, gcd, '\0');
      return -1;
    }

    mpz_gcd(gcd, e, n);
    if (mpz_cmp_ui(gcd, 1) == 0)    // Have we found an e?
      break;

    mpz_nextprime(e, e);            // Bump e to the next prime and try again.
  }

  // There will always be an inverse of e mod phi(n) unless the above code is
  // incorrect, so just do something dumb, like halt execution.
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

void rsa_encrypt(mpz_t c, const mpz_t m, const mpz_t e, const mpz_t n) {
  mpz_powm(c, m, e, n);
}

void rsa_decrypt(mpz_t m, const mpz_t c, const mpz_t d, const mpz_t n) {
  mpz_powm(m, c, d, n);
}
