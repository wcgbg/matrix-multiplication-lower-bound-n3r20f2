#pragma once
#include <algorithm>
#include <array>
#include <initializer_list>
#include <mutex>
#include <random>
#include <vector>

#include <boost/container/static_vector.hpp>
#include <boost/unordered/unordered_flat_map.hpp>
#include <boost/unordered/unordered_flat_set.hpp>
#include <ng-log/logging.h>
#include <tbb/parallel_for.h>

#include "proof_verifier/math_utils.h"
#include "proof_verifier/restrictions.h"
#include "proof_verifier/static_matrix.h"

template <int n0, int n1, int n2> class RestrictionsMap {
public:
  struct Value {
    uint8_t rank = 0;
    uint8_t transpose = 0;
    StaticMatrix<n1> gl_right;
  };
  static_assert(sizeof(Value) == 4);

public:
  explicit RestrictionsMap() {
    static_assert(n0 * n0 < 32);
    static_assert(n1 * n1 < 32);
    static_assert(n0 * n1 < 32);
    LOG(INFO) << "Constructing RestrictionsMap for n0=" << n0 << ", n1=" << n1
              << ", n2=" << n2;
    for (uint32_t d00 = 0; d00 < (uint32_t(1) << (n0 * n0)); ++d00) {
      StaticMatrix<n0> mat00(d00);
      int r00 = mat00.Rank();
      if (r00 == n0) {
        full_rank_matrices_n0_.push_back(mat00);
      }
    }
    std::shuffle(full_rank_matrices_n0_.begin(), full_rank_matrices_n0_.end(),
                 gen_);
    for (uint32_t d11 = 0; d11 < (uint32_t(1) << (n1 * n1)); ++d11) {
      StaticMatrix<n1> mat11(d11);
      int r11 = mat11.Rank();
      if (r11 == n1) {
        full_rank_matrices_n1_.push_back(mat11);
      }
      inverse_n1_.push_back(mat11.Inversed());
    }
    for (uint32_t d00 = 0; d00 < (uint32_t(1) << (n0 * n0)); ++d00) {
      StaticMatrix<n0> mat00(d00);
      for (uint32_t d01 = 0; d01 < (uint32_t(1) << (n0 * n1)); ++d01) {
        StaticMatrix<n0, n1> mat01(d01);
        multiplication_table_n001_.push_back(mat00 * mat01);
      }
    }
    for (uint32_t d01 = 0; d01 < (uint32_t(1) << (n0 * n1)); ++d01) {
      StaticMatrix<n0, n1> mat01(d01);
      for (uint32_t d11 = 0; d11 < (uint32_t(1) << (n1 * n1)); ++d11) {
        StaticMatrix<n1> mat11(d11);
        multiplication_table_n011_.push_back(mat01 * mat11);
      }
    }
    for (uint32_t d01 = 0; d01 < (uint32_t(1) << (n0 * n1)); ++d01) {
      StaticMatrix<n0, n1> mat01(d01);
      m_to_transpose_n01_.push_back(mat01.Transposed());
    }
    LOG(INFO) << "Done";
  }

  virtual ~RestrictionsMap() { Clear(); }

  // Set the rank for all canonical forms of the given restrictions.
  void Set(const Restrictions<n0, n1> &restrictions, int rank) {
    CHECK_GE(rank, 0);
    CHECK_LT(rank, 256);
    Restrictions<n0, n1> new_restrictions;
    for (const StaticMatrix<n1> &gl_right : full_rank_matrices_n1_) {
      for (bool transpose : {false, true}) {
        if (n0 != n1 || n1 != n2 || n0 != n2) {
          if (transpose) {
            break;
          }
        }
        new_restrictions.clear();
        new_restrictions.reserve(restrictions.size());
        for (const auto &restriction : restrictions) {
          StaticMatrixData<n0, n1> new_restriction_data = restriction;
          if (transpose) {
            DCHECK_EQ(n0, n1);
            DCHECK_EQ(n1, n2);
            if constexpr (n0 == n1 && n1 == n2) {
              new_restriction_data =
                  m_to_transpose_n01_[new_restriction_data].Data();
            }
          }
          new_restriction_data =
              Times011(StaticMatrix<n0, n1>(new_restriction_data), gl_right)
                  .Data();
          new_restrictions.push_back(new_restriction_data);
        }
        int new_restrictions_rank =
            GaussJordanElimination(n0 * n1, &new_restrictions);
        CHECK_EQ(new_restrictions_rank, restrictions.size());
        new_restrictions.shrink_to_fit();
        size_t map_index = boost::hash_value(new_restrictions) % kNumOfMaps;
        std::lock_guard<std::mutex> lock(mutexes_[map_index]);
        maps_[map_index][new_restrictions] =
            Value{static_cast<uint8_t>(rank), transpose,
                  inverse_n1_[gl_right.Data()]};
      }
    }
  }

  // Get the rank for the given restrictions.
  int Get(const Restrictions<n0, n1> &restrictions,
          bool *nullable_transpose = nullptr,
          StaticMatrix<n0> *nullable_gl_left = nullptr,
          StaticMatrix<n1> *nullable_gl_right = nullptr) const {
    Restrictions<n0, n1> new_restrictions;
    for (const StaticMatrix<n0> &gl_left : full_rank_matrices_n0_) {
      new_restrictions.clear();
      new_restrictions.reserve(restrictions.size());
      for (const auto &restriction : restrictions) {
        StaticMatrixData<n0, n1> new_restriction_data =
            Times001(gl_left, StaticMatrix<n0, n1>(restriction)).Data();
        new_restrictions.push_back(new_restriction_data);
      }
      int new_restrictions_rank =
          GaussJordanElimination(n0 * n1, &new_restrictions);
      new_restrictions.erase(new_restrictions.begin(),
                             new_restrictions.end() - new_restrictions_rank);
      size_t map_index = boost::hash_value(new_restrictions) % kNumOfMaps;
      auto it = maps_[map_index].find(new_restrictions);
      if (it != maps_[map_index].end()) {
        const Value &entry = it->second;
        if (nullable_gl_left) {
          *nullable_gl_left = gl_left;
        }
        if (nullable_transpose) {
          *nullable_transpose = entry.transpose;
        }
        if (nullable_gl_right) {
          *nullable_gl_right = entry.gl_right;
        }
        return entry.rank;
      }
    }
    LOG(FATAL) << "Restrictions not found: "
               << RestrictionsToString<n0, n1>(restrictions);
  }

  void Clear() {
    tbb::parallel_for(0, kNumOfMaps, [this](int i) {
      std::lock_guard<std::mutex> lock(mutexes_[i]);
      maps_[i].clear();
    });
  }

private:
  static constexpr int kNumOfMaps = 997;
  StaticMatrix<n0, n1> Times001(const StaticMatrix<n0> &m00,
                                const StaticMatrix<n0, n1> &m01) const {
    return multiplication_table_n001_[(m00.Data() << (n0 * n1)) | m01.Data()];
  }
  StaticMatrix<n0, n1> Times011(const StaticMatrix<n0, n1> &m01,
                                const StaticMatrix<n1> &m11) const {
    return multiplication_table_n011_[(m01.Data() << (n1 * n1)) | m11.Data()];
  }
  StaticMatrix<n0, n1> Times0011(const StaticMatrix<n0> &m00,
                                 const StaticMatrix<n0, n1> &m01,
                                 const StaticMatrix<n1> &m11) const {
    return Times011(Times001(m00, m01), m11);
  }

  std::mt19937_64 gen_;
  std::vector<StaticMatrix<n0>> full_rank_matrices_n0_;
  std::vector<StaticMatrix<n1>> full_rank_matrices_n1_;
  std::vector<StaticMatrix<n1>> inverse_n1_;
  std::vector<StaticMatrix<n1, n0>> m_to_transpose_n01_; // n0*n1 -> n1*n0
  std::vector<StaticMatrix<n0, n1>> multiplication_table_n001_;
  std::vector<StaticMatrix<n0, n1>> multiplication_table_n011_;
  std::array<boost::unordered_flat_map<Restrictions<n0, n1>, Value>, kNumOfMaps>
      maps_;
  std::array<std::mutex, kNumOfMaps> mutexes_;
};
