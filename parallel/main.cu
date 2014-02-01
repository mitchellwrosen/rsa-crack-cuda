#include <stdio.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <string.h>
#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>

#include "integer.h"

#define BLOCK_SIZE 32
#define NUM_KEYS 200000
#define BIT_MATRIX_LENGTH 625000 // 200000 * 200000 / 2 / 32

/* Mmaps the file with name |filename| and returns the char array, and puts
 * its size in memory into |length|. */
char *MapFile(const char *filename, size_t *length) {
  int fd = open(filename, O_RDONLY);
  if (fd < 0) {
    perror("failed to open file"); exit(1);
  }

  struct stat s;
  int status = fstat(fd, &s);
  if (status) {
    perror("failed to get file stat"); exit(1);
  }

  *length = s.st_size;
  char *mapped = (char *) mmap(NULL, *length, PROT_READ, MAP_SHARED, fd, 0);
  close(fd);

  if (mapped == MAP_FAILED) {
    perror("map failed"); exit(1);
  }

  return mapped;
}

/* host function that calls kernel, returns bit matrix*/
void factorKeys(const integer *h_array, int *h_matrix) {
  integer *d_array;
  int *d_matrix;

  int grid_dim = (BLOCK_SIZE - 1) / 32 + 1;

  cudaMalloc(&d_array, NUM_KEYS * sizeof(integer));
  cudaMalloc(&d_matrix, BIT_MATRIX_LENGTH * sizeof(int));

  cudaMemcpy(d_array, h_array, NUM_KEYS * sizeof(integer), cudaMemcpyHostToDevice);

  dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid(grid_dim, grid_dim);

  //kernel

  cudaMemcpy(h_matrix, d_matrix, BIT_MATRIX_LENGTH * sizeof(int), cudaMemcpyDeviceToHost);
}
