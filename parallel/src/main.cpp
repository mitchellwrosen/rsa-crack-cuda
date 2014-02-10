#include <stdio.h>
#include <stdint.h>
#include <gmp.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "cuda_utils.h"
#include "integer.h"
#include "rsa.h"

#define TILE_DIM 512
#define BLKS_PER_TILE TILE_DIM/BLOCK_DIM
#define WARP_DIM 32
#define NUM_TILES(N) (N-1)/TILE_DIM+1

using namespace std;

static int X_MASKS[BLOCK_DIM] = { 0x1111, 0x2222, 0x4444, 0x8888 };
static int Y_MASKS[BLOCK_DIM] = { 0x000F, 0x00F0, 0x0F00, 0xF000 };

static int *cracked = NULL;
static int crackedLen = 0;

inline bool checkIfCrackedAlready(int n) {
  for (int i = 0; i < crackedLen; ++i) {
    if (n == cracked[i])
      return true;
  }

  return false;
}

/**
 * Helper function to read keys from txt file using GMP library
 */
inline void getKeysFromFile(const char *filename, integer *keys, int num) {
  FILE *f = fopen(filename, "r");
  mpz_t n;
  mpz_init(n);

  for (int i = 0; i < num; ++i) {
    mpz_inp_str(n, f, 10);
    mpz_export(keys[i].ints, NULL, 1, sizeof(uint32_t), 0, 0, n);
  }

  fclose(f);
}

inline void printUsage(char* progname);
inline void init(integer** keys, uint16_t** notCoprime, integer** d_keys, uint16_t** d_notCoprime, const char* filename, const int numKeys);
void calculatePrivateKeys(integer *array, uint16_t* notCoprime, int tileRow, int tileCol, FILE *stream);

int main(int argc, char **argv) {
  if (argc < 3)
    printUsage(argv[0]);

  int numKeys = atoi(argv[2]);

  integer *keys, *d_keys;
  uint16_t *notCoprime, *d_notCoprime;

  cracked = (int *) malloc(numKeys * sizeof(int));

  init(&keys, &notCoprime, &d_keys, &d_notCoprime, argv[1], numKeys);

  dim3 gridDim(TILE_DIM / BLOCK_DIM, TILE_DIM / BLOCK_DIM);
  dim3 blockDim(WARP_DIM, BLOCK_DIM, BLOCK_DIM);
  int num_tiles = NUM_TILES(numKeys);

  FILE *outputStream = argc == 4 ? fopen(argv[3], "w") : stdout;

  /**
   * Group key pairs into square tiles to fit GPU resource usage limits.
   * For each tile: clear out the bit matrix, run GCD kernel, calculate private keys.
   */
  for (int i = 0; i < num_tiles; ++i) {
    for (int j = i; j < num_tiles; ++j) {
      cudaSafe(cudaMemset(d_notCoprime, 0, BLKS_PER_TILE * BLKS_PER_TILE * sizeof(uint16_t)));

      cudaWrapper(gridDim, blockDim, d_keys, d_notCoprime, i, j, TILE_DIM, numKeys);
      cudaSafe(cudaPeekAtLastError());
      cudaSafe(cudaDeviceSynchronize());

      cudaSafe(cudaMemcpy(notCoprime,
                          d_notCoprime,
                          BLKS_PER_TILE * BLKS_PER_TILE * sizeof(uint16_t),
                          cudaMemcpyDeviceToHost));

      calculatePrivateKeys(keys, notCoprime, i, j, outputStream);
    }
  }

  cudaSafe(cudaFree(d_keys));
  cudaSafe(cudaFree(d_notCoprime));

  free(keys);
  free(notCoprime);
  free(cracked);

  return 0;
}

/**
 * Perform boilerplate initialization: read keys from file, allocate
 * host/device keys, host/device notCoprime, memcpy keys to device.
 */
inline void init(integer** keys,
            uint16_t** notCoprime,
            integer** d_keys,
            uint16_t** d_notCoprime,
            const char* filename,
            const int numKeys) {
  *keys = (integer*) malloc(numKeys * sizeof(integer));
  getKeysFromFile(filename, *keys, numKeys);

  *notCoprime = (uint16_t*) malloc(BLKS_PER_TILE * BLKS_PER_TILE * sizeof(uint16_t));

  cudaSafe(cudaMalloc((void **) d_keys, numKeys * sizeof(integer)));
  cudaSafe(cudaMemcpy(*d_keys, *keys, numKeys * sizeof(integer), cudaMemcpyHostToDevice));

  cudaSafe(cudaMalloc((void **) d_notCoprime, BLKS_PER_TILE * BLKS_PER_TILE * sizeof(uint16_t)));
}

void calculatePrivateKeys(integer* keys, uint16_t* notCoprime, int tileRow, int tileCol, FILE *stream) {
  mpz_t n1, n2, p, q1, q2, d1, d2;
  mpz_inits(n1, n2, p, q1, q2, d1, d2, NULL);

  for (int i = 0; i < BLKS_PER_TILE; ++i) {
    for (int j = 0; j < BLKS_PER_TILE; ++j) {
      uint16_t notCoprimeBlock = notCoprime[i * BLKS_PER_TILE + j];

      if (notCoprimeBlock) {
        for (int y = 0; y < BLOCK_DIM; ++y) {
          if (notCoprimeBlock & Y_MASKS[y]) {
            for (int x = 0; x < BLOCK_DIM; ++x) {
              if (notCoprimeBlock & Y_MASKS[y] & X_MASKS[x]) {
                int n1Ndx = tileRow * TILE_DIM + i * BLOCK_DIM + y;
                int n2Ndx = tileCol * TILE_DIM + j * BLOCK_DIM + x;
                bool crackedN1 = checkIfCrackedAlready(n1Ndx);
                bool crackedN2 = checkIfCrackedAlready(n2Ndx);

                if (!crackedN1 || !crackedN2) {
                  mpz_import(n1, N, 1, sizeof(uint32_t), 0, 0, keys[n1Ndx].ints);
                  mpz_import(n2, N, 1, sizeof(uint32_t), 0, 0, keys[n2Ndx].ints);

                  mpz_gcd(p, n1, n2);

                  if (!crackedN1) {
                    mpz_divexact(q1, n1, p);
                    rsa_compute_d(d1, n1, p, q1);
                    mpz_out_str(stream, 10, n1);
                    fputc(':', stream);
                    mpz_out_str(stream, 10, d1);
                    fputc('\n', stream);

                    cracked[crackedLen++] = n1Ndx;
                  }

                  if (!crackedN2) {
                    mpz_divexact(q2, n2, p);
                    rsa_compute_d(d2, n2, p, q2);
                    mpz_out_str(stream, 10, n2);
                    fputc(':', stream);
                    mpz_out_str(stream, 10, d2);
                    fputc('\n', stream);

                    cracked[crackedLen++] = n2Ndx;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

inline void printUsage(char* progname) {
  cerr << "Usage: " << progname << " keyfile numkeys [outputfile]" << endl;
  exit(EXIT_FAILURE);
}
