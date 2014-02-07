#ifndef INTEGER_H_
#define INTEGER_H_

#include <stdint.h>

#define N 32 // Set integer width at compile time to avoid other inefficiencies

// use struct to avoid pointer decay issues later
typedef struct integer { int32_t ints[N]; } integer;

#endif  // INTEGER_H_
