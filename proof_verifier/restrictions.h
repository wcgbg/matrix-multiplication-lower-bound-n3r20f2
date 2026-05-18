#pragma once

#include <bit>
#include <cstdint>
#include <cstring>
#include <limits>
#include <sstream>
#include <string>

#include <boost/container/static_vector.hpp>
#include <boost/functional/hash.hpp>
#include <ng-log/logging.h>

#include "proof_verifier/math_utils.h"
#include "proof_verifier/restricted_mm.pb.h"
#include "proof_verifier/static_matrix.h"
#include "proof_verifier/tensor_utils.h"

template <int n0, int n1>
using Restrictions = std::vector<StaticMatrixData<n0, n1>>;

template <int n0, int n1>
Restrictions<n0, n1>
RestrictionsFromCompactString(const std::string &compact_string) {
  static_assert(std::is_same_v<uint16_t, StaticMatrixData<n0, n1>>);
  static_assert(std::endian::native == std::endian::little);
  CHECK_EQ(compact_string.size() % sizeof(StaticMatrixData<n0, n1>), 0);
  Restrictions<n0, n1> restrictions(compact_string.size() /
                                    sizeof(StaticMatrixData<n0, n1>));
  std::memcpy(restrictions.data(), compact_string.data(),
              compact_string.size());
  return restrictions;
}

template <int n0, int n1>
std::string
RestrictionsToCompactString(const Restrictions<n0, n1> &restrictions) {
  static_assert(std::is_same_v<uint16_t, StaticMatrixData<n0, n1>>);
  static_assert(std::endian::native == std::endian::little);
  return std::string(reinterpret_cast<const char *>(restrictions.data()),
                     restrictions.size() * sizeof(StaticMatrixData<n0, n1>));
}

template <int n0, int n1>
int NumRestrictions(const pb::RestrictedMM &rmm) {
  CHECK_EQ(rmm.compact_restrictions().size() %
               sizeof(StaticMatrixData<n0, n1>),
           0);
  return rmm.compact_restrictions().size() /
         sizeof(StaticMatrixData<n0, n1>);
}

template <int n0, int n1>
Restrictions<n0, n1> RestrictionsFromProto(const pb::RestrictedMM &rmm) {
  return RestrictionsFromCompactString<n0, n1>(rmm.compact_restrictions());
}

template <int n0, int n1>
std::string RestrictionsToString(const Restrictions<n0, n1> &restrictions) {
  if (restrictions.empty()) {
    return "EMPTY";
  }
  std::ostringstream oss;
  bool is_first = true;
  for (const auto &restriction : restrictions) {
    if (!is_first) {
      oss << ',';
    }
    oss << StaticMatrix<n0, n1>(restriction).ToString();
    is_first = false;
  }
  return oss.str();
}

/*
Apply restrictions to a tensor (F_2 only).
In particular, for each non-zero restriction, substitute the last variable in
the restriction with the sum of the other variables. For example, consider the
restriction a01+a10+a11=0. Move the last variable to the other side:
a11=a01+a10. Then, substitute a11 with a01+a10 in the tensor.
*/
template <int n0, int n1, int n2>
Tensor<n0, n1, n2>
ApplyRestrictionsToTensor(const Restrictions<n0, n1> &restrictions,
                          const Tensor<n0, n1, n2> &tensor) {
  Tensor<n0, n1, n2> result = tensor;

  for (StaticMatrixData<n0, n1> restriction_data : restrictions) {
    if (restriction_data == 0) {
      continue;
    }
    // the index to the most significant bit of the restriction
    int ij_pivot = std::numeric_limits<StaticMatrixData<n0, n1>>::digits - 1 -
                   std::countl_zero(restriction_data);
    boost::container::static_vector<int, n0 * n1> ij_others;
    for (int ij = 0; ij < n0 * n1; ++ij) {
      if (ij != ij_pivot && ((restriction_data >> ij) & 1)) {
        ij_others.push_back(ij);
      }
    }

    for (int jk = 0; jk < n1 * n2; ++jk) {
      for (int ki = 0; ki < n2 * n0; ++ki) {
        uint8_t v = result[ij_pivot][jk][ki];
        for (int ij_other : ij_others) {
          result[ij_other][jk][ki] ^= v;
        }
        result[ij_pivot][jk][ki] = 0;
      }
    }
  }
  return result;
}

// Fast hash for Restrictions (a small std::vector of integral DataType).
// Reads the buffer 8 bytes at a time.
template <typename MatrixDataType = uint16_t> struct RestrictionsHash {
  static_assert(std::is_integral_v<MatrixDataType>);
  size_t operator()(const std::vector<MatrixDataType> &r) const noexcept {
    const uint8_t *data = reinterpret_cast<const uint8_t *>(r.data());
    size_t bytes = r.size() * sizeof(MatrixDataType);
    uint64_t h = bytes;
    while (bytes >= 8) {
      uint64_t chunk;
      std::memcpy(&chunk, data, 8);
      h = (chunk + (h << 6)) + (0x9e3779b9 + (h >> 2));
      data += 8;
      bytes -= 8;
    }
    if (bytes > 0) {
      uint64_t chunk = 0;
      std::memcpy(&chunk, data, bytes);
      h = (chunk + (h << 6)) + (0x9e3779b9 + (h >> 2));
    }
    return h;
  }
};

// [gl_left * transpose(restriction) * gl_right]
template <int n0, int n1, int n2>
Restrictions<n0, n1>
TransformRestrictions(const Restrictions<n0, n1> &restrictions, bool transpose,
                      const StaticMatrix<n0> &gl_left,
                      const StaticMatrix<n1> &gl_right) {
  Restrictions<n0, n1> new_restrictions;
  new_restrictions.reserve(restrictions.size());
  for (const StaticMatrixData<n0, n1> &restriction : restrictions) {
    StaticMatrix<n0, n1> m(restriction);
    if (transpose) {
      CHECK_EQ(n0, n1);
      CHECK_EQ(n1, n2);
      if constexpr (n0 == n1 && n1 == n2) {
        m = m.Transposed();
      }
    }
    m = gl_left * m * gl_right;
    new_restrictions.push_back(m.Data());
  }
  int new_restrictions_rank =
      GaussJordanElimination(n0 * n1, &new_restrictions);
  new_restrictions.erase(new_restrictions.begin(),
                         new_restrictions.end() - new_restrictions_rank);
  return new_restrictions;
}

// [transpose(gl_left * restriction * gl_right)]
template <int n0, int n1, int n2>
Restrictions<n0, n1>
TransformRestrictions(const Restrictions<n0, n1> &restrictions,
                      const StaticMatrix<n0> &gl_left,
                      const StaticMatrix<n1> &gl_right, bool transpose) {
  Restrictions<n0, n1> new_restrictions;
  new_restrictions.reserve(restrictions.size());
  for (const StaticMatrixData<n0, n1> &restriction : restrictions) {
    StaticMatrix<n0, n1> m(restriction);
    m = gl_left * m * gl_right;
    if (transpose) {
      CHECK_EQ(n0, n1);
      CHECK_EQ(n1, n2);
      if constexpr (n0 == n1 && n1 == n2) {
        m = m.Transposed();
      }
    }
    new_restrictions.push_back(m.Data());
  }
  int new_restrictions_rank =
      GaussJordanElimination(n0 * n1, &new_restrictions);
  new_restrictions.erase(new_restrictions.begin(),
                         new_restrictions.end() - new_restrictions_rank);
  return new_restrictions;
}
