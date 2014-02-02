#include <stdlib.h>
#include "kernel.h"

#define FILENAME "keys.txt"

int main(int argc, char **args) {
  integer *array = (integer *) malloc(NUM_KEYS * sizeof(integer));

  int numTiles = (NUM_KEYS - 1) / TILE_DIM + 1;

  int32_t *bitMatrix = (int32_t *) malloc(NUM_KEYS * BIT_MATRIX_WIDTH * sizeof(int32_t));

  // parse file

  host_factorKeys(array, bitMatrix, numTiles);

  // calculate private keys

  // write keys to result.out

  free(array);
  free(bitMatrix);

  return 0;
}
