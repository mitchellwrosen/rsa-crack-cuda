#include <stdio.h>
#include <stdint.h>
#include <gmp.h>
#include <iostream>

#include "integer.h"

#define NUM_KEYS 200000

#define FILENAME "keys.txt"
#define OUTPUT_FILENAME "result.out"

using namespace std;

inline void calculatePrivateKeys(integer *array, int32_t *bitMatrix, int bitMatrixWidth, int tileRow, int tileCol) {
  for (int i = 0; i < TILE_DIM; ++i) {
    int32_t *row = bitMatrix + i * bitMatrixWidth;
    for (int j = i; j < TILE_DIM; ++j) {
      int col = j / 32;
      int offset = j % 32;

      if (row[col] >> offset & 1) {
        int globalRow = tileRow * TILE_DIM + i;
        int globalCol = tileCol * TILE_DIM + j;

        // calculate key
      }
    }
  }
}

inline void getKeysFromFile(const char *filename, integer *array, int num) {
  FILE *f = fopen(filename, "r");
  mpz_t n;
  mpz_init(n);

  for (int i = 0; i < num; ++i) {
    mpz_inp_str(n, f, 10);
    mpz_export(array[i], NULL, 1, sizeof(int32_t), 1, 0, n);
  }

  fclose(f);
}

int main(int argc, char **args) {
  integer *array = new integer[NUM_KEYS];
  int32_t *bitMatrix;

  int numTiles = allocateGPU(array, &bitMatrix, NUM_KEYS);
  int bitMatrixWidth = (TILE_DIM - 1) / 32 + 1;

  getKeysFromFile(FILENAME, array, NUM_KEYS);

  for (int i = 0; i < numTiles; ++i) {
    for (int j = i; j < numTiles; ++j) {
      factorKeys(array, bitMatrix, NUM_KEYS, i, j);
      calculatePrivateKeys(array, bitMatrix, bitMatrixWidth, i, j);
    }
  }

  freeGPU();
  free(array);
  free(bitMatrix);

  return 0;
}
