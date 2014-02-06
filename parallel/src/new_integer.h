#ifndef INTEGER_H_
#define INTEGER_H_

#include <stdint.h>

#define N 32 // Set integer width at compile time to avoid other inefficiencies


// This will replace integer.h and integer.cu

typedef int32_t integer[N];

#endif  // INTEGER_H_
