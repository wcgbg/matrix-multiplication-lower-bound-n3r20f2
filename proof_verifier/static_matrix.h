/*
A matrix over F_2 of size n0 x n1, represented by the bits of an unsigned
integer.
*/

#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <ostream>
#include <random>
#include <sstream>
#include <string>

#include <ng-log/logging.h>

// An n0 x n1 matrix over F_2, represented by an unsigned integer.
template <int n0, int n1 = n0> class StaticMatrix {
public:
  using DataType = uint16_t;
  static_assert(n0 > 0 && n1 > 0 &&
                n0 * n1 <= std::numeric_limits<DataType>::digits);

  // Field characteristic (F_2)
  static constexpr int p() { return 2; }

  // Default constructor: zero matrix
  StaticMatrix() : data_(0) {}

  // Constructor from unsigned integer
  explicit StaticMatrix(DataType data) : data_(data) {}

  static StaticMatrix Identity() {
    static_assert(n0 == n1, "Identity only for square matrices");
    StaticMatrix result;
    for (int i = 0; i < n0; ++i) {
      result.Set(i, i, 1);
    }
    return result;
  }
  // The dense string should have n0*n1 numbers. All the other characters are
  // ignored. For example, "[101,110,001]" is valid.
  static StaticMatrix FromString(const std::string &dense_string) {
    DataType result{};
    int idx = 0;
    for (char c : dense_string) {
      if (c == '0' || c == '1') {
        CHECK_LT(idx, n0 * n1) << dense_string;
        if (c == '1') {
          result |= (DataType(1) << idx);
        }
        ++idx;
      }
    }
    CHECK_EQ(idx, n0 * n1) << dense_string;
    return StaticMatrix(result);
  }

  static StaticMatrix Random(std::mt19937_64 *gen) {
    static_assert(n0 * n1 < 32);
    std::uniform_int_distribution<uint32_t> dist(0, (uint32_t(1) << (n0 * n1)) -
                                                        1);
    return StaticMatrix(dist(*gen));
  }

  // Get element at position (i, j)
  uint8_t Get(int i, int j) const {
    DCHECK_GE(i, 0);
    DCHECK_LT(i, n0);
    DCHECK_GE(j, 0);
    DCHECK_LT(j, n1);
    int bit_pos = i * n1 + j;
    return (data_ >> bit_pos) & 1;
  }

  // Set element at position (i, j)
  void Set(int i, int j, uint8_t value) {
    DCHECK_GE(i, 0);
    DCHECK_LT(i, n0);
    DCHECK_GE(j, 0);
    DCHECK_LT(j, n1);
    DCHECK_LT(value, 2);
    int bit_pos = i * n1 + j;
    if (value) {
      data_ |= (1U << bit_pos);
    } else {
      data_ &= ~(1U << bit_pos);
    }
  }

  // Plus (XOR) operation
  StaticMatrix Plus(const StaticMatrix &other) const {
    return StaticMatrix(data_ ^ other.data_);
  }

  // Check if matrix is zero
  bool IsZero() const { return data_ == 0; }

  // Check if matrix is identity
  bool IsIdentity() const {
    static_assert(n0 == n1, "Identity only for square matrices");
    return *this == Identity();
  }

  // Compute rank over F_2 using Gaussian elimination
  int Rank() const {
    std::array<DataType, n0> rows = {0};
    for (int i = 0; i < n0; ++i) {
      rows[i] = (data_ >> (i * n1)) & ((DataType(1) << n1) - 1);
    }

    int rank = 0;
    int pivot_col = 0;

    for (int row = 0; row < n0 && pivot_col < n1; ++row) {
      int pivot_row = -1;
      for (int r = row; r < n0; ++r) {
        if (rows[r] & (DataType(1) << pivot_col)) {
          pivot_row = r;
          break;
        }
      }

      if (pivot_row == -1) {
        ++pivot_col;
        --row;
        continue;
      }

      if (pivot_row != row) {
        std::swap(rows[row], rows[pivot_row]);
      }

      for (int r = 0; r < n0; ++r) {
        if (r != row && (rows[r] & (DataType(1) << pivot_col))) {
          rows[r] ^= rows[row];
        }
      }

      ++rank;
      ++pivot_col;
    }

    return rank;
  }

  // Get raw data
  DataType Data() const { return data_; }

  // Print the matrix in a compact format, like [101,110,001]
  std::string ToString() const {
    std::ostringstream oss;
    oss << '[';
    for (int i = 0; i < n0; ++i) {
      for (int j = 0; j < n1; ++j) {
        oss << (Get(i, j) ? '1' : '0');
      }
      if (i < n0 - 1) {
        oss << ',';
      }
    }
    oss << ']';
    return oss.str();
  }

  // Matrix multiplication over F_2
  template <int n2>
  StaticMatrix<n0, n2> operator*(const StaticMatrix<n1, n2> &other) const {
    StaticMatrix<n0, n2> result;
    for (int i = 0; i < n0; ++i) {
      for (int j = 0; j < n2; ++j) {
        uint8_t value = 0;
        for (int k = 0; k < n1; ++k) {
          value ^= (Get(i, k) & other.Get(k, j));
        }
        result.Set(i, j, value);
      }
    }
    return result;
  }

  // Compute the inverse matrix over F_2 (square matrices only)
  // Returns the inverse if the matrix is invertible (rank == n0)
  // Returns zero matrix if not invertible
  StaticMatrix Inversed() const {
    static_assert(n0 == n1, "Inverse only for square matrices");
    std::array<DataType, n0> rows = {0};
    std::array<DataType, n0> identity = {0};

    for (int i = 0; i < n0; ++i) {
      rows[i] = (data_ >> (i * n1)) & ((DataType(1) << n1) - 1);
      identity[i] = (DataType(1) << i);
    }

    for (int col = 0; col < n0; ++col) {
      int pivot_row = -1;
      for (int r = col; r < n0; ++r) {
        if (rows[r] & (DataType(1) << col)) {
          pivot_row = r;
          break;
        }
      }

      if (pivot_row == -1) {
        return StaticMatrix(0);
      }

      if (pivot_row != col) {
        std::swap(rows[col], rows[pivot_row]);
        std::swap(identity[col], identity[pivot_row]);
      }

      for (int r = 0; r < n0; ++r) {
        if (r != col && (rows[r] & (DataType(1) << col))) {
          rows[r] ^= rows[col];
          identity[r] ^= identity[col];
        }
      }
    }

    StaticMatrix result;
    for (int i = 0; i < n0; ++i) {
      for (int j = 0; j < n1; ++j) {
        if (identity[i] & (DataType(1) << j)) {
          result.Set(i, j, 1);
        }
      }
    }
    return result;
  }

  // Compute the transpose matrix
  StaticMatrix<n1, n0> Transposed() const {
    StaticMatrix<n1, n0> result;
    for (int i = 0; i < n0; ++i) {
      for (int j = 0; j < n1; ++j) {
        result.Set(j, i, Get(i, j));
      }
    }
    return result;
  }

  // Comparison operators
  bool operator==(const StaticMatrix &other) const {
    return data_ == other.data_;
  }
  bool operator!=(const StaticMatrix &other) const {
    return data_ != other.data_;
  }
  bool operator<(const StaticMatrix &other) const {
    return data_ < other.data_;
  }
  bool operator>(const StaticMatrix &other) const {
    return data_ > other.data_;
  }
  bool operator<=(const StaticMatrix &other) const {
    return data_ <= other.data_;
  }
  bool operator>=(const StaticMatrix &other) const {
    return data_ >= other.data_;
  }

private:
  DataType data_ = 0;
};

template <int n0, int n1 = n0>
std::ostream &operator<<(std::ostream &os, const StaticMatrix<n0, n1> &m) {
  os << m.ToString();
  return os;
}

template <int n0, int n1 = n0>
using StaticMatrixData = typename StaticMatrix<n0, n1>::DataType;

namespace std {
template <int n0, int n1> struct hash<StaticMatrix<n0, n1>> {
  size_t operator()(const StaticMatrix<n0, n1> &m) const {
    return std::hash<typename StaticMatrix<n0, n1>::DataType>{}(m.Data());
  }
};
} // namespace std
