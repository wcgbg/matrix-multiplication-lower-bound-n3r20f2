#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <ng-log/logging.h>

std::string GetBacktrackingProofRootDir(const std::string &proto_path);

std::string GetBacktrackingProofPath(const std::string &bt_proof_root_dir,
                                     int index, bool create_dir = false);

struct BacktrackingProof {
  void Save(const std::string &path) const;
  static BacktrackingProof Load(const std::string &path);
  void Append(uint8_t dfs_restrictions_size, uint32_t mask, bool transpose,
              uint16_t gl_left, uint16_t gl_right) {
    dfs_restrictions_size_array.push_back(dfs_restrictions_size);
    mask_array.push_back(mask);
    transpose_array.push_back(transpose);
    gl_left_array.push_back(gl_left);
    gl_right_array.push_back(gl_right);
  }
  void Append(const BacktrackingProof &other);
  void Check() const {
    size_t n = dfs_restrictions_size_array.size();
    CHECK_EQ(n, mask_array.size());
    CHECK_EQ(n, transpose_array.size());
    CHECK_EQ(n, gl_left_array.size());
    CHECK_EQ(n, gl_right_array.size());
  }
  size_t Size() const { return dfs_restrictions_size_array.size(); }
  void Reserve(size_t n) {
    dfs_restrictions_size_array.reserve(n);
    mask_array.reserve(n);
    transpose_array.reserve(n);
    gl_left_array.reserve(n);
    gl_right_array.reserve(n);
  }

  std::vector<uint8_t> dfs_restrictions_size_array;
  std::vector<uint32_t> mask_array;
  std::vector<bool> transpose_array;
  std::vector<uint16_t> gl_left_array;
  std::vector<uint16_t> gl_right_array;
};
