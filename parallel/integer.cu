#include "integer.h"

__device__ void shiftR(integer output, integer input) {
   for (int i = 0; i < N-1; ++i)
      output[i] = (input[i] >> 1) | (input[i+1] << 31)
   output[N-1] = input[N-1] >> 1;
}

__device__ void shiftL(integer output, integer input) {
   for (int i = N-1; i > 0; --i)
      output[i] = (input[i] << 1) | (input[i-1] >> 31)
   output[0] = input[0] << 1;
}

__device__ bool geq(integer a, integer b) {
   for (int i = N-1; i >= 0; --i) {
      if (a[i] < b[i])
         return false;
   }
   return true;
}

__device__ void sub(integer result, integer a, integer b) {
   bool underflow = 0;
   for (int i = 0; i < N; ++i) {
      result[i] = a[i] - b[i] - underflow;
      underflow = result[i] < 0 ? 1 : 0;
   }
}
