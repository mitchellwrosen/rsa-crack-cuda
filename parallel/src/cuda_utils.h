#ifndef CUDA_UTILS_H_
#define CUDA_UTILS_H_

#define cudaSafe(ans) { cudaAssert((ans), __FILE__, __LINE__); }

inline void cudaAssert(cudaError_t code, char *file, int line, bool abort=true) {
  if (code != cudaSuccess) {
    fprintf(stderr,"cudaAssert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort)
      exit(code);
  }
}

#endif  // CUDA_UTILS_H_
