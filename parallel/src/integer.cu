#include "integer.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

__device__ void gcd(volatile uint32_t *x, volatile uint32_t *y);
__device__ void shiftR1(volatile uint32_t *x);
__device__ void shiftL1(volatile uint32_t *x);
__device__ int geq(volatile uint32_t *x, volatile uint32_t *y);
__device__ void cuSubtract(volatile uint32_t *x, volatile uint32_t *y, volatile uint32_t *z);

/* kernel */
__global__ void cuda_factorKeys(const integer *keys, uint32_t *notCoprime, size_t pitch, int tileRow, int tileCol, int tileDim, int numKeys) {
  __shared__ volatile uint32_t x[BLOCK_DIM][BLOCK_DIM][32];
  __shared__ volatile uint32_t y[BLOCK_DIM][BLOCK_DIM][32];

  int keyX = tileCol * tileDim + blockIdx.x * BLOCK_DIM + threadIdx.y;
  int keyY = tileRow * tileDim + blockIdx.y * BLOCK_DIM + threadIdx.z;

  if (keyX < numKeys && keyY < numKeys && keyX < keyY) {
    x[threadIdx.y][threadIdx.z][threadIdx.x] = keys[keyX].ints[threadIdx.x];
    y[threadIdx.y][threadIdx.z][threadIdx.x] = keys[keyY].ints[threadIdx.x];

    gcd(x[threadIdx.y][threadIdx.z], y[threadIdx.y][threadIdx.z]);

    if (threadIdx.x == 31) {
      y[threadIdx.y][threadIdx.z][threadIdx.x] -= 1;

      if (__any(y[threadIdx.y][threadIdx.z][threadIdx.x])) {
        /* int notCoprimeKeyX = keyX - tileCol * tileDim; */
        /* int notCoprimeKeyY = keyY - tileRow * tileDim; */

        /* uint32_t *notCoprimeRow = (uint32_t *) ((char *) notCoprime + notCoprimeKeyX * pitch); */
        /* int notCoprimeCol = notCoprimeKeyY / 32; */
        /* int notCoprimeOffset = notCoprimeKeyY % 32; */

        /* notCoprimeRow[notCoprimeCol] |= 1 << notCoprimeOffset; */
      }
    }
  }
}

void cuda_wrapper(dim3 gridDim, dim3 blockDim, integer* d_keys, uint32_t* d_notCoprime,
    size_t pitch, int tileRow, int tileCol, int tileDim, int numKeys) {
      cuda_factorKeys<<<gridDim, blockDim>>>(d_keys, d_notCoprime,
          pitch, tileRow, tileCol, tileDim, numKeys);
}

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
