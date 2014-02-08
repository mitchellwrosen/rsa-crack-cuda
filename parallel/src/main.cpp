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

#define FILENAME "../256-keys.txt"
#define OUTPUT_FILENAME "result.out"

#define NUM_KEYS 256
#define TILE_DIM 512
#define NUM_TILES (NUM_KEYS-1)/TILE_DIM+1
#define BIT_MATRIX_WIDTH (TILE_DIM-1)/32+1

using namespace std;

extern void cuda_wrapper(dim3 gridDim, dim3 blockDim, integer* d_keys, uint32_t* d_notCoprime,
    size_t pitch, int tileRow, int tileCol, int tileDim, int numKeys);

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

size_t init(integer** keys, integer** d_keys, uint32_t** d_notCoprime);
void calculatePrivateKeys(integer *array, const BitMatrix& notCoprime, int tileRow, int tileCol);

int main(int argc, char **args) {
  integer *keys, *d_keys;
  uint32_t *d_notCoprime;
  BitMatrix notCoprime(TILE_DIM);

  size_t pitch = init(&keys, &d_keys, &d_notCoprime);

  dim3 gridDim(TILE_DIM / BLOCK_DIM, TILE_DIM / BLOCK_DIM);
  dim3 blockDim(32, BLOCK_DIM, BLOCK_DIM);
  int num_tiles = NUM_TILES;

  for (int i = 0; i < num_tiles; ++i) {
    for (int j = i; j < num_tiles; ++j) {
      cudaSafe(cudaMemset2D(d_notCoprime, pitch, 0, BIT_MATRIX_WIDTH * sizeof(uint32_t), TILE_DIM));

      cuda_wrapper(gridDim, blockDim, d_keys, d_notCoprime, pitch, i, j, TILE_DIM, NUM_KEYS);
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
size_t init(integer** keys, integer** d_keys, uint32_t** d_notCoprime) {
  *keys = (integer*) malloc(NUM_KEYS * sizeof(integer));
  getKeysFromFile(FILENAME, *keys, NUM_KEYS);

  cudaSafe(cudaMalloc((void **) d_keys, NUM_KEYS * sizeof(integer)));
  cudaSafe(cudaMemcpy(*d_keys, *keys, NUM_KEYS * sizeof(integer), cudaMemcpyHostToDevice));

  size_t pitch;
  cudaSafe(cudaMallocPitch((void **) d_notCoprime, &pitch, BIT_MATRIX_WIDTH * sizeof(uint32_t), TILE_DIM));

  return pitch;
}

void calculatePrivateKeys(integer* keys, const BitMatrix& notCoprime, int tileRow, int tileCol) {
  for (int i = 0; i < notCoprime.size()-1; ++i) {
    for (int j = i; j < notCoprime.size(); ++j) {
      if (notCoprime.bitSet(i, j)) {
        integer key1 = keys[tileRow * TILE_DIM + i];
        integer key2 = keys[tileCol * TILE_DIM + j];

        // calculate key
      }
    }
  }
}
