#pragma once

#include <bit>
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
Restrictions<n0, n1> RestrictionsFromProto(const pb::RestrictedMM &rmm) {
  Restrictions<n0, n1> restrictions;
  for (const auto &restriction_proto : rmm.restriction()) {
    CHECK_EQ(restriction_proto.a_size(), n0 * n1);
    StaticMatrix<n0, n1> restriction;
    for (int i = 0; i < n0; ++i) {
      for (int j = 0; j < n1; ++j) {
        int value = restriction_proto.a(i * n1 + j);
        CHECK_GE(value, 0);
        CHECK_LT(value, 2) << "coefficient must be in F_2";
        restriction.Set(i, j, static_cast<uint8_t>(value));
      }
    }
    restrictions.push_back(restriction.Data());
  }
  return restrictions;
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

template <int n0, int n1> struct RestrictionsHash {
  size_t operator()(const Restrictions<n0, n1> &r) const {
    return boost::hash_value(r);
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
