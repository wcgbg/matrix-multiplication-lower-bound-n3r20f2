#pragma once

#include <array>
#include <cstdint>
#include <format>
#include <sstream>
#include <string>
#include <vector>

#include <boost/algorithm/string.hpp>
#include <ng-log/logging.h>

template <int n0, int n1, int n2>
using Tensor =
    std::array<std::array<std::array<uint8_t, n2 * n0>, n1 * n2>, n0 * n1>;

template <int n> using SquareTensor = Tensor<n, n, n>;

template <int n0, int n1, int n2>
std::string TensorToSparseString(const Tensor<n0, n1, n2> &tensor) {
  // Convert a tensor to its sparse string representation, e.g.,
  // a00*b10*c01 + a01*b00*c01 + ... (F2), or 2*a00*b10*c01 + ... (F3,F5,F7).
  // If the tensor is all zeros, return "0".
  std::ostringstream oss;
  bool first_term = true;
  for (int aij = 0; aij < n0 * n1; ++aij) {
    int ai = aij / n1;
    int aj = aij % n1;
    for (int bij = 0; bij < n1 * n2; ++bij) {
      int bi = bij / n2;
      int bj = bij % n2;
      for (int cij = 0; cij < n2 * n0; ++cij) {
        int ci = cij / n0;
        int cj = cij % n0;
        uint8_t v = tensor[aij][bij][cij];
        if (v == 0) {
          continue;
        }
        CHECK_LT(v, 10);
        if (!first_term) {
          oss << " + ";
        }
        if (v != 1) {
          oss << int(v) << '*';
        }
        oss << std::format("a{}{}*b{}{}*c{}{}", ai, aj, bi, bj, ci, cj);
        first_term = false;
      }
    }
  }
  if (first_term) {
    return "0";
  }
  return oss.str();
}

template <int n0, int n1, int n2>
std::string TensorToDenseString(const Tensor<n0, n1, n2> &tensor) {
  std::ostringstream oss;
  for (int ij = 0; ij < n0 * n1; ++ij) {
    for (int jk = 0; jk < n1 * n2; ++jk) {
      for (int ki = 0; ki < n2 * n0; ++ki) {
        oss << int(tensor[ij][jk][ki]);
      }
      oss << ',';
    }
    oss << '\n';
  }
  return oss.str();
}

// `text` is like a00*b10*c01 + a01*b00*c01 (F2), or 2*a00*b10*c01 + ...
// (F3,F5,F7).
template <int n0, int n1, int n2>
Tensor<n0, n1, n2> SparseStringToTensor(const std::string &text) {
  Tensor<n0, n1, n2> tensor = {};
  if (text == "0") {
    return tensor;
  }
  std::vector<std::string> terms;
  boost::split(terms, text, boost::is_any_of("+"));
  for (auto &term : terms) {
    boost::algorithm::trim(term);
    int coef = 1;
    if (term.size() == 13) {
      coef = term[0] - '0';
      CHECK_LT(coef, 10) << term;
      CHECK_EQ(term[1], '*');
      term = term.substr(2);
    }
    CHECK_EQ(term.size(), 11) << term;
    CHECK_EQ(term[0], 'a');
    int ai = term[1] - '0';
    int aj = term[2] - '0';
    CHECK_GE(ai, 0);
    CHECK_LT(ai, n0) << term;
    CHECK_GE(aj, 0);
    CHECK_LT(aj, n1) << term;
    CHECK_EQ(term[3], '*');
    CHECK_EQ(term[4], 'b');
    int bi = term[5] - '0';
    int bj = term[6] - '0';
    CHECK_GE(bi, 0);
    CHECK_LT(bi, n1) << term;
    CHECK_GE(bj, 0);
    CHECK_LT(bj, n2) << term;
    CHECK_EQ(term[7], '*');
    CHECK_EQ(term[8], 'c');
    int ci = term[9] - '0';
    int cj = term[10] - '0';
    CHECK_GE(ci, 0);
    CHECK_LT(ci, n2) << term;
    CHECK_GE(cj, 0);
    CHECK_LT(cj, n0) << term;
    CHECK_LT(coef, 10) << term;
    int aij = ai * n1 + aj;
    int bij = bi * n2 + bj;
    int cij = ci * n0 + cj;
    CHECK_EQ(tensor[aij][bij][cij], 0);
    tensor[aij][bij][cij] = static_cast<uint8_t>(coef);
  }
  return tensor;
}

// Parse tensor from the string format output by TensorToDenseString.
// The string has (n0*n1)*(n1*n2)*(n2*n0) digits.
template <int n0, int n1, int n2>
Tensor<n0, n1, n2> DenseStringToTensor(const std::string &text) {
  Tensor<n0, n1, n2> tensor = {};
  int i = 0;
  int j = 0;
  int k = 0;
  for (char c : text) {
    if (c < '0' || c > '9') {
      continue;
    }
    CHECK_LT(i, n0 * n1) << "Too many digits in the string: " << text;
    uint8_t d = static_cast<uint8_t>(c - '0');
    CHECK_LT(d, 10);
    tensor[i][j][k] = d;
    k++;
    if (k == n2 * n0) {
      k = 0;
      j++;
      if (j == n1 * n2) {
        j = 0;
        i++;
      }
    }
  }
  CHECK_EQ(i, n0 * n1) << text;
  CHECK_EQ(j, 0) << text;
  CHECK_EQ(k, 0) << text;
  return tensor;
}

template <int n0, int n1, int n2>
Tensor<n1, n2, n0> CyclicTranspose(const Tensor<n0, n1, n2> &tensor) {
  Tensor<n1, n2, n0> result = {};
  for (int ij = 0; ij < n0 * n1; ++ij) {
    for (int jk = 0; jk < n1 * n2; ++jk) {
      for (int ki = 0; ki < n2 * n0; ++ki) {
        result[jk][ki][ij] = tensor[ij][jk][ki];
      }
    }
  }
  return result;
}

template <int n0, int n1, int n2>
Tensor<n0, n1, n2> MatrixMultiplicationTensor() {
  Tensor<n0, n1, n2> tensor = {};
  for (int i = 0; i < n0; ++i) {
    for (int j = 0; j < n1; ++j) {
      for (int k = 0; k < n2; ++k) {
        int ij = i * n1 + j;
        int jk = j * n2 + k;
        int ki = k * n0 + i;
        tensor[ij][jk][ki] = 1;
      }
    }
  }
  return tensor;
}
