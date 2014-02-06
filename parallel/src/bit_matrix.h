#ifndef BIT_MATRIX_H_
#define BIT_MATRIX_H_

class BitMatrix {
 public:
  BitMatrix(int size);
  virtual ~BitMatrix();

  void clear();

  bool bitSet(int row, int col) const;

  int size() const      { return size_; }
  int pitch() const     { return pitch_;  }
  int32_t* data() const { return data_;   }

 private:
  int size_;

  int actual_width_; // in words
  size_t pitch_;     // in bytes

  int32_t* data_;
}

#endif  // BIT_MATRIX_H_

