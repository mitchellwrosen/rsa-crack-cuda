#include <stdio.h>
#include "kernel.h"

#define cudaSafe(ans) { cudaAssert((ans), __FILE__, __LINE__); }

/* wrapper assert function to check for cuda errors */
inline void cudaAssert(cudaError_t code, char *file, int line, bool abort=true) {
  if (code != cudaSuccess) {
    fprintf(stderr,"cudaAssert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort)
      exit(code);
  }
}

/* kernel */
__global__ void cuda_factorKeys(const integer *array, int32_t *bitMatrix, size_t pitch, int tile_row, int tile_col) {
  int col = tile_col * TILE_DIM + blockIdx.x * BLOCK_DIM + threadIdx.x;
  int row = tile_row * TILE_DIM + blockIdx.y * BLOCK_DIM + threadIdx.y;

  if (row < col && row < NUM_KEYS && col < NUM_KEYS) {
    if (!integer_coprime(array[col], array[row])) {
      int32_t *bitMatrix_row = (int32_t *) ((char *) bitMatrix + row * pitch);
      int bitMatrix_col = col / 32;
      int bitMatrix_offset = col % 32;

      bitMatrix_row[bitMatrix_col] |= (1 << 31) >> bitMatrix_offset;
    }
  }
}

/* host function that calls kernel, returns bit matrix */
void host_factorKeys(const integer *h_array, int32_t *h_bitMatrix, const int numTiles) {
  // allocate array of moduli on device
  integer *d_array;
  cudaSafe(cudaMalloc(&d_array, NUM_KEYS * sizeof(integer)));
  cudaSafe(cudaMemcpy(d_array, h_array, NUM_KEYS * sizeof(integer), cudaMemcpyHostToDevice));

  // allocate (and initialize to 0) bit matrix
  int32_t *d_bitMatrix;
  size_t d_pitch;
  size_t h_pitch = BIT_MATRIX_DIM * sizeof(int32_t);
  cudaSafe(cudaMallocPitch(&d_bitMatrix, &d_pitch, h_pitch, BIT_MATRIX_DIM));
  cudaSafe(cudaMemset2D(d_bitMatrix, d_pitch, 0, h_pitch, BIT_MATRIX_DIM));

  dim3 threads(BLOCK_DIM, BLOCK_DIM);
  dim3 grid(TILE_DIM / BLOCK_DIM, TILE_DIM / BLOCK_DIM);

  for (int i = 0; i < numTiles; ++i) {
    for (int j = i; j < numTiles; ++j) {
      cuda_factorKeys<<<grid, threads>>>(d_array, d_bitMatrix, d_pitch, i, j);
    }
  }

  cudaSafe(cudaMemcpy2D(h_bitMatrix, h_pitch, d_bitMatrix, d_pitch, d_pitch, BIT_MATRIX_DIM, cudaMemcpyDeviceToHost));
  cudaSafe(cudaFree(d_array));
  cudaSafe(cudaFree(d_bitMatrix));
}

