#include "dynamic_matrix.h"

#include <sstream>

#include <ng-log/logging.h>

DynamicMatrix::DynamicMatrix(int n, int m) : n_(n), m_(m), data_(n * m, 0) {}

void DynamicMatrix::ResizeRows(int n) {
  n_ = n;
  data_.resize(n * m_);
}

std::string DynamicMatrix::ToString() const {
  std::ostringstream oss;
  oss << '[';
  for (int i = 0; i < n_; ++i) {
    for (int j = 0; j < m_; ++j) {
      oss << ((*this)(i, j) ? '1' : '0');
    }
    if (i < n_ - 1) {
      oss << ',';
    }
  }
  oss << ']';
  return oss.str();
}

DynamicMatrix DynamicMatrix::Plus(const DynamicMatrix &other) const {
  CHECK_EQ(n_, other.n_) << "Plus: row count mismatch";
  CHECK_EQ(m_, other.m_) << "Plus: column count mismatch";
  DynamicMatrix result(n_, m_);
  for (int i = 0; i < n_; ++i) {
    for (int j = 0; j < m_; ++j) {
      result(i, j) = ((*this)(i, j) ^ other(i, j)) & 1;
    }
  }
  return result;
}

bool DynamicMatrix::IsZero() const {
  for (int i = 0; i < n_; ++i) {
    for (int j = 0; j < m_; ++j) {
      if ((*this)(i, j) != 0) {
        return false;
      }
    }
  }
  return true;
}

int DynamicMatrix::Rank() const {
  if (n_ == 0 || m_ == 0) {
    return 0;
  }
  // Copy to row-major 2D structure for Gaussian elimination.
  std::vector<std::vector<uint8_t>> rows(n_, std::vector<uint8_t>(m_));
  for (int i = 0; i < n_; ++i) {
    for (int j = 0; j < m_; ++j) {
      rows[i][j] = (*this)(i, j);
    }
  }

  int rank = 0;
  int pivot_col = 0;

  for (int row = 0; row < n_ && pivot_col < m_; ++row) {
    // Find pivot row (row with 1 in current pivot column)
    int pivot_row = -1;
    for (int r = row; r < n_; ++r) {
      if (rows[r][pivot_col]) {
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

    // Eliminate below and above the pivot
    for (int r = 0; r < n_; ++r) {
      if (r != row && rows[r][pivot_col]) {
        for (int c = 0; c < m_; ++c) {
          rows[r][c] ^= rows[row][c];
        }
      }
    }

    ++rank;
    ++pivot_col;
  }

  return rank;
}
