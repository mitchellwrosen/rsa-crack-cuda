#include "integer.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

__device__ void gcd(volatile uint32_t *x, volatile uint32_t *y);
__device__ void shiftR1(volatile uint32_t *x);
__device__ void shiftL1(volatile uint32_t *x);
__device__ int geq(volatile uint32_t *x, volatile uint32_t *y);
__device__ void cuSubtract(volatile uint32_t *x, volatile uint32_t *y, volatile uint32_t *z);

/**
 * Kernel function.
 * See "PARIS: A Parallel RSA-Prime Inspection Tool" by Joseph White
 */
__global__ void cuda_factorKeys(const integer *keys, uint16_t *notCoprime, size_t pitch, int tileRow, int tileCol, int tileDim, int numKeys) {
  /* shared memory for keys */
  __shared__ volatile uint32_t y[BLOCK_DIM][BLOCK_DIM][32];
  __shared__ volatile uint32_t z[BLOCK_DIM][BLOCK_DIM][32];

  /* determine key indexes */
  int keyX = tileCol * tileDim + blockIdx.x * BLOCK_DIM + threadIdx.y;
  int keyY = tileRow * tileDim + blockIdx.y * BLOCK_DIM + threadIdx.z;

  /* only continue w/ warp if we need to to run this comparison */
  if (keyX < numKeys && keyY < numKeys && keyX < keyY) {
    /* each thread loads its corresponding int into shared memory */
    y[threadIdx.y][threadIdx.z][threadIdx.x] = keys[keyX].ints[threadIdx.x];
    z[threadIdx.y][threadIdx.z][threadIdx.x] = keys[keyY].ints[threadIdx.x];

    /* run gcd */
    gcd(y[threadIdx.y][threadIdx.z], z[threadIdx.y][threadIdx.z]);

    if (threadIdx.x == 31) {
      /* turn gcd=1 to 0 */
      z[threadIdx.y][threadIdx.z][threadIdx.x] -= 1;

      /* check if any ints in the warp are > 0, which means gcd > 1
       * update notCoprime */
      if (__any(z[threadIdx.y][threadIdx.z][threadIdx.x])) {
        int notCoprimeBlockNdx = blockIdx.y * gridDim.x + blockIdx.x;
        notCoprime[notCoprimeBlockNdx] |= 1 << (threadIdx.y * BLOCK_DIM + threadIdx.z);
        printf("cuda %d %d\n", keyX + 1, keyY + 1);
      }
    }
  }
}

void cuda_wrapper(dim3 gridDim, dim3 blockDim, integer* d_keys, uint16_t* d_notCoprime,
    size_t pitch, int tileRow, int tileCol, int tileDim, int numKeys) {
      cuda_factorKeys<<<gridDim, blockDim>>>(d_keys, d_notCoprime,
          pitch, tileRow, tileCol, tileDim, numKeys);
}

/**
 * Binary GCD algo
 */
__device__ void gcd(volatile uint32_t *x, volatile uint32_t *y) {
  int tid = threadIdx.x;

  while (__any(x[tid])) {
    while ((x[31] & 1) == 0)
      shiftR1(x);

    while ((y[31] & 1) == 0)
      shiftR1(y);

    if (geq(x, y)) {
      cuSubtract(x, y, x);
      shiftR1(x);
    }
    else {
      cuSubtract(y, x, y);
      shiftR1(y);
    }
  }
}

__device__ void shiftR1(volatile uint32_t *x) {
  int tid = threadIdx.x;
  uint32_t prevX = tid ? x[tid-1] : 0;
  x[tid] = (x[tid] >> 1) | (prevX << 31);
}

__device__ void shiftL1(volatile uint32_t *x) {
  int tid = threadIdx.x;
  uint32_t nextX = tid != 31 ? x[tid+1] : 0;
  x[tid] = (x[tid] << 1) | (nextX >> 31);
}

__device__ int geq(volatile uint32_t *x, volatile uint32_t *y) {
  __shared__ unsigned int pos[BLOCK_DIM][BLOCK_DIM];
  int tid = threadIdx.x;

  if (tid == 0)
    pos[threadIdx.y][threadIdx.z] = 31;

  if (x[tid] != y[tid])
    atomicMin(&pos[threadIdx.y][threadIdx.z], tid);

  return x[pos[threadIdx.y][threadIdx.z]] >= y[pos[threadIdx.y][threadIdx.z]];
}

__device__ void cuSubtract(volatile uint32_t *x, volatile uint32_t *y, volatile uint32_t *z) {
  __shared__ unsigned char s_borrow[BLOCK_DIM][BLOCK_DIM][32];
  unsigned char *borrow = s_borrow[threadIdx.y][threadIdx.z];
  int tid = threadIdx.x;

  if (tid == 0)
    borrow[31] = 0;

  uint32_t t;
  t = x[tid] - y[tid];

  if(tid)
    borrow[tid - 1] = (t > x[tid]);

  while (__any(borrow[tid])) {
    if (borrow[tid])
      t--;

    if (tid)
      borrow[tid - 1] = (t == 0xFFFFFFFFu && borrow[tid]);
  }

  z[tid] = t;
}
