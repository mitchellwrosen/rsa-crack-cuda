#ifndef INTEGER_H_
#define INTEGER_H_

#include <stdint.h>

#define BLOCK_DIM 4

#define N 32 // Set integer width at compile time to avoid other inefficiencies

typedef struct integer { uint32_t ints[N]; } integer;

void cudaWrapper(dim3 gridDim, dim3 blockDim, integer* d_keys, uint16_t* d_notCoprime, int tileRow, int tileCol, int tileDim, int numKeys);

#endif  // INTEGER_H_
