#include <stdio.h>
#include <stdint.h>
#include <gmp.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "bit_matrix.h"
#include "cuda_utils.h"
#include "integer.h"
#include "rsa.h"

#define OUTPUT_FILENAME "result.out"

#define TILE_DIM 512
#define NUM_TILES(N) (N-1)/TILE_DIM+1
#define BIT_MATRIX_WIDTH (TILE_DIM-1)/32+1

using namespace std;

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
inline size_t init(integer** keys, integer** d_keys, uint32_t** d_notCoprime, const char* filename, const int numKeys);
inline void calculatePrivateKeys(integer *array, const BitMatrix& notCoprime, int tileRow, int tileCol);

int main(int argc, char **argv) {
  if (argc != 3)
    printUsage(argv[0]);

  int numKeys = atoi(argv[2]);

  integer *keys, *d_keys;
  uint32_t *d_notCoprime;
  BitMatrix notCoprime(TILE_DIM);

  size_t pitch = init(&keys, &d_keys, &d_notCoprime, argv[1], numKeys);

  dim3 gridDim(TILE_DIM / BLOCK_DIM, TILE_DIM / BLOCK_DIM);
  dim3 blockDim(32, BLOCK_DIM, BLOCK_DIM);
  int num_tiles = NUM_TILES(numKeys);

  /**
   * Group key pairs into square tiles to fit GPU resource usage limits.
   * For each tile: clear out the bit matrix, run GCD kernel, calculate private keys.
   */
  for (int i = 0; i < num_tiles; ++i) {
    for (int j = i; j < num_tiles; ++j) {
      cudaSafe(cudaMemset2D(d_notCoprime, pitch, 0, BIT_MATRIX_WIDTH * sizeof(uint32_t), TILE_DIM));

      cuda_wrapper(gridDim, blockDim, d_keys, d_notCoprime, pitch, i, j, TILE_DIM, numKeys);
      cudaSafe(cudaPeekAtLastError());
      cudaSafe(cudaDeviceSynchronize());

      cudaSafe(cudaMemcpy2D(notCoprime.data(),                   // dst
                            notCoprime.pitch(),                  // dst pitch
                            d_notCoprime,                        // src
                            pitch,                               // src pitch
                            BIT_MATRIX_WIDTH * sizeof(uint32_t),  // width
                            TILE_DIM,                            // height
                            cudaMemcpyDeviceToHost));            // kind

      calculatePrivateKeys(keys, notCoprime, i, j);
    }
  }

  cudaSafe(cudaFree(d_keys));
  cudaSafe(cudaFree(d_notCoprime));

  free(keys);

  return 0;
}

/**
 * Perform boilerplate initialization: read keys from file, allocate
 * host/device keys, device notPrime, memcpy keys to device. Return the pitch
 * of device bit matrix.
 */
inline size_t init(integer** keys, integer** d_keys, uint32_t** d_notCoprime, const char* filename, const int numKeys) {
  *keys = (integer*) malloc(numKeys * sizeof(integer));
  getKeysFromFile(filename, *keys, numKeys);

  cudaSafe(cudaMalloc((void **) d_keys, numKeys * sizeof(integer)));
  cudaSafe(cudaMemcpy(*d_keys, *keys, numKeys * sizeof(integer), cudaMemcpyHostToDevice));

  size_t pitch;
  cudaSafe(cudaMallocPitch((void **) d_notCoprime, &pitch, BIT_MATRIX_WIDTH * sizeof(uint32_t), TILE_DIM));

  return pitch;
}

inline void calculatePrivateKeys(integer* keys, const BitMatrix& notCoprime, int tileRow, int tileCol) {
  mpz_t n1, n2, p, q1, q2, d1, d2;
  mpz_inits(n1, n2, p, q1, q2, d1, d2, '\0');

  for (int i = 0; i < notCoprime.size()-1; ++i) {
    for (int j = i; j < notCoprime.size(); ++j) {
      if (notCoprime.bitSet(i, j)) {
        mpz_import(n1, N, 1, sizeof(uint32_t), 0, 0, keys[tileRow * TILE_DIM + i].ints);
        mpz_import(n2, N, 1, sizeof(uint32_t), 0, 0, keys[tileCol * TILE_DIM + j].ints);

        mpz_gcd(p, n1, n2);
        mpz_divexact(q1, n1, p);
        mpz_divexact(q2, n2, p);
        rsa_compute_d(d1, n1, p, q1);
        rsa_compute_d(d2, n2, p, q2);
        // output d1,d2
      }
    }
  }
}

inline void printUsage(char* progname) {
  cerr << "Usage: " << progname << " keyfile numkeys" << endl;
  exit(EXIT_FAILURE);
}
