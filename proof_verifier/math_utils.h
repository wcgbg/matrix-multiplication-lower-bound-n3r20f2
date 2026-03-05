#pragma once

#include <ng-log/logging.h>

// Perform Gauss-Jordan elimination into Reduced Row Echelon Form (RREF).
// `(*matrix)[i]` is a row of length bitwidth. Row is an unsigned integer type.
// `*matrix` represents a matrix of size `matrix->size()` x `bitwidth` over F_2.
// The row order and column order are both reversed to make (matrix[0],
// matrix[1], ...]) lexicographically minimal.
// Returns number of non-zero rows in the RREF.
template <typename VectorOfUnsignedInts>
int GaussJordanElimination(int bitwidth, VectorOfUnsignedInts *matrix) {
  using Row = std::decay_t<decltype(*matrix->data())>;
  static_assert(std::is_integral_v<Row>, "Row must be an integral type");
  static_assert(std::is_unsigned_v<Row>,
                "Row must be an unsigned integer type");

  if (matrix->empty() || bitwidth <= 0) {
    return 0;
  }

  int rank = 0;
  int current_row = static_cast<int>(matrix->size()) - 1;
  int pivot_col = bitwidth - 1;

  // Create a mask for the bitwidth
  Row bitwidth_mask = (bitwidth < static_cast<int>(sizeof(Row) * 8))
                          ? ((Row(1) << bitwidth) - 1)
                          : ~Row(0);

  // Gauss-Jordan elimination (reversed row and column order)
  while (current_row >= 0 && pivot_col >= 0) {
    // Find a pivot row (row with 1 in current column)
    int pivot_row = -1;
    for (int r = current_row; r >= 0; --r) {
      if ((*matrix)[r] & (Row(1) << pivot_col)) {
        pivot_row = r;
        break;
      }
    }

    if (pivot_row == -1) {
      // No pivot found in this column, try previous column
      --pivot_col;
      continue;
    }

    // Swap rows if needed
    if (pivot_row != current_row) {
      std::swap((*matrix)[current_row], (*matrix)[pivot_row]);
    }

    // Eliminate above and below the pivot (Gauss-Jordan: eliminate in both
    // directions)
    for (int r = 0; r < static_cast<int>(matrix->size()); ++r) {
      if (r != current_row && ((*matrix)[r] & (Row(1) << pivot_col))) {
        (*matrix)[r] ^= (*matrix)[current_row];
        // Mask to ensure we only keep bits within bitwidth
        (*matrix)[r] &= bitwidth_mask;
      }
    }

    ++rank;
    --current_row;
    --pivot_col;
  }

  return rank;
}

template <typename VectorOfUnsignedInts>
bool IsLinearIndependent(int bitwidth, VectorOfUnsignedInts matrix) {
  return GaussJordanElimination(bitwidth, &matrix) == matrix.size();
}
