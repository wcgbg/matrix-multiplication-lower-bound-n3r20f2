#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

// Matrix over F_2 of size n x m.
class DynamicMatrix {
public:
  DynamicMatrix(int n, int m);
  template <size_t n, size_t M>
  explicit DynamicMatrix(const std::array<std::array<uint8_t, M>, n> &data);

  void ResizeRows(int n);

  // Like "[1001,1101,0001]".
  std::string ToString() const;

  uint8_t operator()(int i, int j) const { return data_[Index(i, j)]; }
  uint8_t &operator()(int i, int j) { return data_[Index(i, j)]; }

  DynamicMatrix Plus(const DynamicMatrix &other) const;

  bool IsZero() const;

  // Compute rank using Gaussian elimination
  int Rank() const;

  int rows() const { return n_; }
  int cols() const { return m_; }

private:
  int Index(int i, int j) const { return i * m_ + j; }

  int n_ = 0;
  int m_ = 0;
  std::vector<uint8_t> data_;
};

template <size_t n, size_t M>
DynamicMatrix::DynamicMatrix(const std::array<std::array<uint8_t, M>, n> &data)
    : n_(n), m_(M), data_(n * M) {
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < M; ++j) {
      (*this)(static_cast<int>(i), static_cast<int>(j)) = data[i][j];
    }
  }
}
