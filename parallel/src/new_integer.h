#ifndef INTEGER_H_
#define INTEGER_H_

#include <stdint.h>

#define BLOCK_DIM 32
#define TILE_DIM 20000

#define N 32 // Set integer width at compile time to avoid other inefficiencies


// This will replace integer.h and integer.cu

typedef int32_t integer[N];

void factorKeys(integer *h_array, int32_t *h_bitMatrix, const int numKeys, const int tileX, const int tileY);

int allocateGPU(integer *h_array, int32_t **h_bitMatrix, const int numKeys);

void freeGPU();

#endif  // INTEGER_H_
