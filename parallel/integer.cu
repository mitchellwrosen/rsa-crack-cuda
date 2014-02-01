#include "integer.h"

#include <string.h>

// Misc. initialization functions
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

// Number theoretic functions.
__device__ bool integer_coprime(const integer a, const integer b) {
   integer gcd;
   integer_gcd(gcd, a, b);        // gcd = gcd(a,b);
   return integer_eqInt(gcd, 1);  // gcd == 1 ?
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
