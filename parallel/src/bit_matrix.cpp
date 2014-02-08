#include "bit_matrix.h"

#include <string.h>

BitMatrix::BitMatrix(int size)
    : size_(size)
    , actual_width_((size-1)/32+1)
    , pitch_(actual_width_ * sizeof(uint32_t))
    , data_((uint32_t*) malloc(size_ * pitch_)) {
  clear();
}

BitMatrix::~BitMatrix() {
  free(data_);
}

void BitMatrix::clear() {
  memset(data_, '\0', size_ * actual_width_ * sizeof(uint32_t));
}

bool BitMatrix::bitSet(int row, int col) const {
  int index = actual_width_ * row  // index into row
            + col/32;              // index into col
  return data_[index] & (1 << (col%32));
}
