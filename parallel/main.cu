#include <stdio.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <string.h>
#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>

#include "integer.h"

#define BLOCK_DIM 32
#define NUM_KEYS 200000

#define TILE_DIM 192

#define FILENAME "keys.txt"

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
__global__ void cuda_factorKeys(const integer *array, bool *h_matrix, int tile_row, int tile_col) {
  int col = blockIdx.x * BLOCK_DIM + threadIdx.x;
  int row = blockIdx.y * BLOCK_DIM + threadIdx.y;

  if (row < NUM_KEYS && col < NUM_KEYS) {
    if (!integer_coprime(array[col], array[row])) {
      // update matrix
    }
  }
}

/* host function that calls kernel, returns bit matrix*/
void host_factorKeys(const integer *h_array, bool **h_matrices, const int num_tiles) {
  int tile_size = TILE_DIM * TILE_DIM * sizeof(bool);
  integer *d_array;

  cudaMalloc(&d_array, NUM_KEYS * sizeof(integer));

  cudaMemcpy(d_array, h_array, NUM_KEYS * sizeof(integer), cudaMemcpyHostToDevice);

  bool **d_matrices;
  size_t pitch;
  cudaMallocPitch(&d_matrices, &pitch, TILE_DIM * sizeof(bool), TILE_DIM);

  dim3 threads(BLOCK_DIM, BLOCK_DIM);
  dim3 grid(TILE_DIM / BLOCK_DIM, TILE_DIM / BLOCK_DIM);

  for (int i = 0; i < num_tiles; ++i) {
    for (int j = i; j < num_tiles; ++j) {
      cuda_factorKeys<<<grid, threads>>>(d_array, d_matrices[i*pitch + j], i, j);
      cudaMemcpy(h_matrices[i*TILE_DIM + j], d_matrices[i*pitch + j], tile_size, cudaMemcpyDeviceToHost);
    }
  }

  cudaFree(d_array);
  cudaFree(d_matrices);
}

int main(int argc, char **args) {
  integer *array = (integer *) malloc(NUM_KEYS * sizeof(integer));

  int num_tiles = (NUM_KEYS - 1) / TILE_DIM + 1;
  int total_num_tiles = num_tiles * num_tiles;

  bool **matrices = (bool **) malloc(total_num_tiles * sizeof(bool *));
  for (int i = 0; i < total_num_tiles; ++i) {
    matrices[i] = (bool *) malloc(TILE_DIM * TILE_DIM * sizeof(bool));
  }

  size_t file_size;

  char *file_string = map_file(FILENAME, &file_size);

  // parse file

  host_factorKeys(array, matrices, num_tiles);

  // calculate private keys

  // write keys to result.out

  munmap(file_string, file_size);

  free(array);

  return 0;
}
