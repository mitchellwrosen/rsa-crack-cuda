#ifndef INTEGER_H_
#define INTEGER_H_

#include <stdint.h>

#define N 512 // Set integer width at compile time to avoid other inefficiencies

typedef int32_t integer[N];

__device__ void shiftR(integer output, integer input);      // output = input >> 1
__device__ void shiftL(integer output, integer input);      // output = input << 1
__device__ bool geq(integer a, integer b);                  // a >= b
__device__ void sub(integer result, integer a, integer b);  // result = a - b

#endif  // INTEGER_H_
