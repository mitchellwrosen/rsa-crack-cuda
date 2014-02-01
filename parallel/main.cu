#include <stdio.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <string.h>
#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>

#include "integer.h"

#define BLOCK_SIZE 32
#define NUM_KEYS 200000

#define FILENAME "keys.txt"

static long int matrix_length = 625000; // 200000 * 200000 / 2 / 32

/* Mmaps the file with name |filename| and returns the char array, and puts
 * its size in memory into |length|. */
char *map_file(const char *filename, size_t *length) {
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

/* kernel */
__global__ void cuda_factorKeys(const integer *array, int32_t *h_matrix) {
  int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;

  if (row < NUM_KEYS && col < NUM_KEYS) {
    if (integer_coprime(array[col], array[row])) {
      // update matrix
    }
  }
}

/* host function that calls kernel, returns bit matrix*/
void host_factorKeys(const integer *h_array, int32_t *h_matrix) {
  integer *d_array;
  int *d_matrix;

  int grid_dim = (BLOCK_SIZE - 1) / 32 + 1;

  cudaMalloc(&d_array, NUM_KEYS * sizeof(integer));
  cudaMalloc(&d_matrix, matrix_length * sizeof(int));

  cudaMemcpy(d_array, h_array, NUM_KEYS * sizeof(integer), cudaMemcpyHostToDevice);

  dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid(grid_dim, grid_dim);

  cuda_factorKeys<<<grid, threads>>>(d_array, d_matrix);

  cudaMemcpy(h_matrix, d_matrix, matrix_length * sizeof(int), cudaMemcpyDeviceToHost);
}

int main(int argc, char **args) {
  integer *array = (integer *) malloc(NUM_KEYS * sizeof(integer));
  int32_t *bit_matrix = (int32_t *) malloc(matrix_length * sizeof(int32_t));
  size_t file_size;

  char *file_string = map_file(FILENAME, &file_size);

  // parse file

  // host_factorKeys(array, bit_matrix);

  // calculate private keys

  // write keys to result.out

  munmap(file_string, file_size);

  free(array);
  free(bit_matrix);

  return 0;
}
