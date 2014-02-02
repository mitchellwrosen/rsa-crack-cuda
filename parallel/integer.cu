#include "integer.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

// like getline(), but don't include the newline in the resulting line, or in
// the return value.
static ssize_t getline_(char** lineptr, size_t* n, FILE* stream) {
   ssize_t numBytes = getline(lineptr, n, stream);
   if ((*lineptr)[numBytes-1] == '\n') {
      (*lineptr)[numBytes-1] = '\0';
      numBytes--;
   }
   return numBytes;
}

// Count the number of bits |num| requires; 0 <= n <= 32. For example, the
// number '5' requires three bits (101). Works by simply iterating over the
// bits from right to left; when the first 1 is encountered, the count is
// returned.
static unsigned int bitCount(int32_t num) {
   unsigned int count = 32;
   unsigned int mask = 1 << 31;
   while (mask) {
      if (num & mask)
         return count;
      mask >>= 1;
      count--;
   }
   return 0;
}

static void packBits(integer output, int* outputIndex_orig, int* bitIndex_orig, int32_t segment) {
   // Copy indices, because they will be manipulated a lot in this function.
   int outputIndex = *outputIndex_orig;
   int bitIndex = *bitIndex_orig;

   int bitsInSegment = bitCount((unsigned int) segment);
   for (int i = 0; i < bitsInSegment; ++i) {
      if (outputIndex >= N) {
         fprintf(stderr, "Integer too large, cannot fit in %d bits\n", N * 32);
         exit(EXIT_FAILURE);
      }

      if (segment & (1 << i))
         output[outputIndex] |= (1 << bitIndex);

      bitIndex += 1;
      if (bitIndex == 32) {
         bitIndex = 0;
         outputIndex++;
      }
   }

   // Copy indices back.
   *outputIndex_orig = outputIndex;
   *bitIndex_orig = bitIndex;
}

///////////////////////////////////////////////////////////////////////////////

// I/O functions
// Read an integer from a stream. Parse nine characters at a time, from right to
// left, packing bits accordingly. Nine characters because it's guaranteed to
// fit in a single int32_t.
void integer_fread(integer *output, char** lineptr, size_t* n, FILE* stream) {
   memset(*output, '\0', sizeof(*output));
   ssize_t numBytes = getline_(lineptr, n, stream);

   int lineIndex = numBytes - 9;  // current index into line
   int outputIndex = 0;           // current index into |output|
   int bitIndex = 0;              // next bit index to pack in |output| at outputIndex

   // Strange condition, but it works: keep subtracting nine from lineIndex 
   // until there are no more characters to parse. It's okay to go negative,
   // say, to -7, which simply means that there are only two characters left.
   // After parsing everything, though, the final index will be -9.
   while (lineIndex != -9) {
      // Last iteration: adjust index to 0 (beginning of the line).
      if (lineIndex < 0)
         lineIndex = 0;

      int32_t segment = atoi(*lineptr + lineIndex);
      packBits(*output, &outputIndex, &bitIndex, segment);

      (*lineptr)[lineIndex] = '\0'; // end-of-string marker for next atoi()
      lineIndex -= 9;         // move index to parse next 9 bytes.
   }
}

// Initialization functions
__device__ void integer_copy(integer output, const integer input) {
   for (int i = 0; i < N; ++i)
      output[i] = input[i];
}

__device__ void integer_fromInt(integer *output, const int32_t input) {
   memset(*output, '\0', sizeof(*output) - sizeof(int32_t)); // Zero all but lowest order byte
   (*output)[N-2] = input;
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
