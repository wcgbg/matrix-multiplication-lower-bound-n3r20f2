#pragma once

#include <algorithm>
#include <array>
#include <initializer_list>
#include <mutex>
#include <random>
#include <vector>

#include <boost/container/static_vector.hpp>
#include <boost/unordered/unordered_flat_set.hpp>
#include <ng-log/logging.h>
#include <tbb/parallel_for.h>

#include "proof_verifier/restrictions.h"

template <int n0, int n1> class RestrictionsSet {
public:
  static_assert(n0 <= 4);
  static_assert(n1 <= 4);
  static_assert(n0 * n0 < 32);
  static_assert(n1 * n1 < 32);
  static_assert(n0 * n1 < 32);

  RestrictionsSet() = default;
  virtual ~RestrictionsSet() { Clear(); }

  void Insert(const Restrictions<n0, n1> &restrictions) {
    size_t shard_index = boost::hash_value(restrictions) % kNumOfShards;
    std::lock_guard<std::mutex> lock(mutexes_[shard_index]);
    sets_[shard_index].insert(restrictions);
  }

  bool ContainsUnsafe(const Restrictions<n0, n1> &restrictions) const {
    size_t shard_index = boost::hash_value(restrictions) % kNumOfShards;
    return sets_[shard_index].contains(restrictions);
  }

  size_t SizeUnsafe() const {
    size_t size = 0;
    for (int i = 0; i < kNumOfShards; ++i) {
      size += sets_[i].size();
    }
    return size;
  }

  void Clear() {
    tbb::parallel_for(0, kNumOfShards, [this](int i) {
      std::lock_guard<std::mutex> lock(mutexes_[i]);
      sets_[i].clear();
    });
  }

private:
  static constexpr int kNumOfShards = 997;

  std::array<boost::unordered_flat_set<Restrictions<n0, n1>>, kNumOfShards>
      sets_;
  std::array<std::mutex, kNumOfShards> mutexes_;
};
