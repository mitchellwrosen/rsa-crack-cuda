#include "integer.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#define cudaSafe(ans) { cudaAssert((ans), __FILE__, __LINE__); }

static integer *d_array;
static int32_t *d_bitMatrix;
static size_t h_pitch;
static size_t d_pitch;
static int bitMatrixWidth;

/* wrapper assert function to check for cuda errors */
inline void cudaAssert(cudaError_t code, char *file, int line, bool abort=true) {
  if (code != cudaSuccess) {
    fprintf(stderr,"cudaAssert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort)
      exit(code);
  }
}

// Initialization functions
__device__ void integer_copy(integer output, const integer input) {
   for (int i = 0; i < N; ++i)
      output[i] = input[i];
}

__device__ void integer_fromInt(integer output, const int32_t input) {
   memset(output, '\0', sizeof(output) - sizeof(int32_t)); // Zero all but lowest order byte
   output[N-2] = input;
}

// Bit manipulation functions.
__device__ void integer_shiftR(integer output, const integer input) {
   for (int i = 0; i < N-1; ++i)
      output[i] = (input[i] >> 1) | (input[i+1] << 31);
   output[N-1] = input[N-1] >> 1;
}

__device__ void integer_shiftL(integer output, const integer input) {
   for (int i = N-1; i > 0; --i)
      output[i] = (input[i] << 1) | (input[i-1] >> 31);
   output[0] = input[0] << 1;
}

// Comparison functions.
__device__ int integer_cmp(const integer a, const integer b) {
   for (int i = N-1; i >= 0; --i) {
      int32_t diff = a[i] - b[i];
      if (diff != 0)
         return diff;
   }
   return 0;
}

__device__ bool integer_eq(const integer a, const integer b) {
   for (int i = N-1; i >= 0; --i) {
      if (a[i] != b[i])
         return false;
   }
   return true;
}

__device__ bool integer_eqInt(const integer a, const int32_t b) {
   for (int i = N-1; i > 0; --i) {
      if (a[i] != 0)
         return false;
   }
   return a[0] == b;
}

__device__ bool integer_neq(const integer a, const integer b) {
   return !integer_eq(a, b);
}

__device__ bool integer_geq(const integer a, const integer b) {
   for (int i = N-1; i >= 0; --i) {
      if (a[i] < b[i])
         return false;
   }
   return true;
}

// Arithmetic functions.
__device__ void integer_sub(integer result, const integer a, const integer b) {
   bool underflow = 0;
   for (int i = 0; i < N; ++i) {
      result[i] = a[i] - b[i] - underflow;
      underflow = result[i] < 0 ? 1 : 0;
   }
}

__device__ void integer_sub_(integer result, integer a, integer b) {
   integer temp;
   integer_sub(temp, a, b);    // temp = a - b
   integer_copy(result, temp); // result = temp
}

__device__ void integer_gcd(integer result, const integer a, const integer b) {
   integer_copy(result, a);

   integer b_copy;
   integer_copy(b_copy, b);

   int cmp = integer_cmp(result, b_copy);
   while (cmp != 0) {
      if (cmp > 0)                              // if a > b
         integer_sub_(result, result, b_copy);  //    a = a - b
      else                                      // else
         integer_sub_(b_copy, b_copy, result);  //    b = b - a
      cmp = integer_cmp(result, b_copy);
   }                                            // return a
}

// Number theoretic functions.
__device__ bool integer_coprime(const integer a, const integer b) {
   integer gcd;
   integer_gcd(gcd, a, b);        // gcd = gcd(a,b);
   return integer_eqInt(gcd, 1);  // gcd == 1 ?
}

/* kernel */
__global__ void cuda_factorKeys(const integer *array, int32_t *bitMatrix, size_t pitch, int tileRow, int tileCol, int numKeys) {
  int col = tileCol * TILE_DIM + blockIdx.x * BLOCK_DIM + threadIdx.x;
  int row = tileRow * TILE_DIM + blockIdx.y * BLOCK_DIM + threadIdx.y;

  /* if (row < col && row < numKeys && col < numKeys) { */
  /*   if (!integer_coprime(array[col], array[row])) { */
  /*     int32_t *bitMatrix_row = (int32_t *) ((char *) bitMatrix + row * pitch); */
  /*     int bitMatrix_col = col / 32; */
  /*     int bitMatrix_offset = col % 32; */

  /*     bitMatrix_row[bitMatrix_col] |= 1 << bitMatrix_offset; */
  /*   } */
  /* } */

  if (row < col && row < numKeys && col < numKeys && row % 10000 == 0 && col % 5000 == 0) {
    int localCol = col - tileCol * TILE_DIM;
    int localRow = row - tileRow * TILE_DIM;

    int32_t *bitMatrix_row = (int32_t *) ((char *) bitMatrix + localRow * pitch);
    int bitMatrix_col = localCol / 32;
    int bitMatrix_offset = localCol % 32;

    bitMatrix_row[bitMatrix_col] |= 1 << bitMatrix_offset;
  }
}

/* host function that calls kernel, returns bit matrix */
void factorKeys(integer *h_array, int32_t *h_bitMatrix, const int numKeys, const int tileRow, const int tileCol) {
  cudaSafe(cudaMemset2D(d_bitMatrix, d_pitch, 0, h_pitch, TILE_DIM));

  dim3 threads(BLOCK_DIM, BLOCK_DIM);
  dim3 grid(TILE_DIM / BLOCK_DIM, TILE_DIM / BLOCK_DIM);

  cuda_factorKeys<<<grid, threads>>>(d_array, d_bitMatrix, d_pitch, tileRow, tileCol, numKeys);
  cudaSafe(cudaPeekAtLastError());
  cudaSafe(cudaDeviceSynchronize());

  cudaSafe(cudaMemcpy2D(h_bitMatrix, h_pitch, d_bitMatrix, d_pitch, h_pitch, TILE_DIM, cudaMemcpyDeviceToHost));
}

int allocateGPU(integer *h_array, int32_t **h_bitMatrix, const int numKeys) {
  cudaSafe(cudaMalloc(&d_array, numKeys * sizeof(integer)));
  cudaSafe(cudaMemcpy(d_array, h_array, numKeys * sizeof(integer), cudaMemcpyHostToDevice));

  bitMatrixWidth = (TILE_DIM - 1) / 32 + 1;
  *h_bitMatrix = (int32_t *) malloc(TILE_DIM * bitMatrixWidth * sizeof(int32_t));
  h_pitch = bitMatrixWidth * sizeof(int32_t);

  cudaSafe(cudaMallocPitch(&d_bitMatrix, &d_pitch, h_pitch, TILE_DIM));
  cudaSafe(cudaMemset2D(d_bitMatrix, d_pitch, 0, h_pitch, TILE_DIM));

  return (numKeys - 1) / TILE_DIM + 1;
}

void freeGPU() {
  cudaSafe(cudaFree(d_array));
  cudaSafe(cudaFree(d_bitMatrix));
}
