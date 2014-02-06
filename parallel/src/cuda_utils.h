#ifndef CUDA_UTILS_H_
#define CUDA_UTILS_H_

#define cudaSafe(ans) { cudaAssert((ans), __FILE__, __LINE__); }

inline void cudaAssert(cudaError_t code, char *file, int line, bool abort=true);

#endif  // CUDA_UTILS_H_
