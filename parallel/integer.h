#ifndef INTEGER_H_
#define INTEGER_H_

#include <stdint.h>

#define N 512 // Set integer width at compile time to avoid other inefficiencies

typedef uint32_t integer[N];

__device__ void shiftR(integer input, integer output);
__device__ void shiftL(integer input, integer output);
__device__ bool geq(integer a, integer b);

#endif  // INTEGER_H_
