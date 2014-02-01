#ifndef INTEGER_H_
#define INTEGER_H_

#include <stdint.h>

#define N 32 // Set integer width at compile time to avoid other inefficiencies

typedef int32_t integer[N];

// Misc. initialization functions
__device__ void integer_copy(integer output, const integer input);     // output = input
__device__ void integer_fromInt(integer output, const int32_t input);  // output = input

// Bit manipulation functions.
__device__ void integer_shiftR(integer output, const integer input);  // output = input >> 1
__device__ void integer_shiftL(integer output, const integer input);  // output = input << 1

// Comparison functions.
__device__ int integer_cmp(const integer a, const integer b);     // a - b -> returns <0, 0, or >0
__device__ bool integer_eq(const integer a, const integer b);     // a == b
__device__ bool integer_eqInt(const integer a, const int32_t b);  // a == b
__device__ bool integer_neq(const integer a, const integer b);    // a != b
__device__ bool integer_geq(const integer a, const integer b);    // a >= b

// Arithmetic functions.
__device__ void integer_sub(integer result, const integer a, const integer b);  // result = a - b
__device__ void integer_sub_(integer result, integer a, integer b); // can support "a = a - b" or "b = a - b"

// Number theoretic functions.
__device__ bool integer_coprime(const integer a, const integer b);              // are a and b coprime?
__device__ void integer_gcd(integer result, const integer a, const integer b);  // result = gcd(a,b)

#endif  // INTEGER_H_
