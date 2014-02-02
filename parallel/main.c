#include "kernel.h"

#define FILENAME "keys.txt"

int main(int argc, char **args) {
  integer *array = (integer *) malloc(NUM_KEYS * sizeof(integer));

  int numTiles = (NUM_KEYS - 1) / TILE_DIM + 1;

  int32_t *bitMatrix = (int32_t *) malloc(BIT_MATRIX_DIM * BIT_MATRIX_DIM * sizeof(int32_t));

  size_t fileSize;
  char *fileString = mapFile(FILENAME, &fileSize);

  // parse file

  host_factorKeys(array, bitMatrix, numTiles);

  // calculate private keys

  // write keys to result.out

  free(array);
  free(bitMatrix);

  return 0;
}
