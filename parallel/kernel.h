#ifndef KERNEL_H_
#define KERNEL_H_

#include "integer.h"

#define NUM_KEYS 200000

#define BLOCK_DIM 32
#define TILE_DIM 5000
#define BIT_MATRIX_WIDTH ((NUM_KEYS - 1) / 32 + 1)

void host_factorKeys(const integer *h_array, int32_t *h_bitMatrix, const int numTiles);

#endif
