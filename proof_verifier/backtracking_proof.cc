#include "proof_verifier/backtracking_proof.h"

#include <filesystem>
#include <format>

#include <boost/iostreams/device/file.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <ng-log/logging.h>

namespace io = boost::iostreams;

std::string GetBacktrackingProofRootDir(const std::string &proto_path) {
  std::string prefix;
  if (proto_path.ends_with(".pb.txt")) {
    prefix = proto_path.substr(0, proto_path.size() - 7);
  } else if (proto_path.ends_with(".pb")) {
    prefix = proto_path.substr(0, proto_path.size() - 3);
  } else {
    LOG(FATAL) << "Unsupported file extension: " << proto_path;
  }
  return prefix + "_bt_proof";
}

std::string GetBacktrackingProofPath(const std::string &bt_proof_root_dir,
                                     int index, bool create_dir) {
  if (bt_proof_root_dir.empty()) {
    return {};
  }
  CHECK_LT(index, 1'000'000);
  std::string dir = std::format("{}/{:03}", bt_proof_root_dir, index / 1000);
  if (create_dir) {
    std::filesystem::create_directories(dir);
  }
  // btp is for backtracking proof
  return std::format("{}/{:03}.btp", dir, index % 1000);
}

void BacktrackingProof::Save(const std::string &path) const {
  if (path.empty()) {
    return;
  }
  Check();
  const uint64_t n = static_cast<uint64_t>(Size());
  io::filtering_ostream out;
  out.push(io::gzip_compressor());
  out.push(io::file_sink(path));
  CHECK(out) << "Failed to open file for writing: " << path;
  out.write(reinterpret_cast<const char *>(&n), sizeof(n));
  out.write(reinterpret_cast<const char *>(dfs_restrictions_size_array.data()),
            n * sizeof(uint8_t));
  out.write(reinterpret_cast<const char *>(mask_array.data()),
            n * sizeof(uint32_t));
  for (size_t i = 0; i < n; ++i) {
    const uint8_t b = transpose_array[i] ? 1 : 0;
    out.write(reinterpret_cast<const char *>(&b), sizeof(b));
  }
  out.write(reinterpret_cast<const char *>(gl_left_array.data()),
            n * sizeof(uint16_t));
  out.write(reinterpret_cast<const char *>(gl_right_array.data()),
            n * sizeof(uint16_t));
  io::close(out);
}

BacktrackingProof BacktrackingProof::Load(const std::string &path) {
  BacktrackingProof proof;
  io::filtering_istream in;
  in.push(io::gzip_decompressor());
  in.push(io::file_source(path));
  CHECK(in) << "Failed to open file for reading: " << path;
  uint64_t n = 0;
  in.read(reinterpret_cast<char *>(&n), sizeof(n));
  proof.dfs_restrictions_size_array.resize(static_cast<size_t>(n));
  proof.mask_array.resize(static_cast<size_t>(n));
  proof.transpose_array.resize(static_cast<size_t>(n));
  proof.gl_left_array.resize(static_cast<size_t>(n));
  proof.gl_right_array.resize(static_cast<size_t>(n));
  in.read(reinterpret_cast<char *>(proof.dfs_restrictions_size_array.data()),
          n * sizeof(uint8_t));
  in.read(reinterpret_cast<char *>(proof.mask_array.data()),
          n * sizeof(uint32_t));
  for (uint64_t i = 0; i < n; ++i) {
    uint8_t b;
    in.read(reinterpret_cast<char *>(&b), sizeof(b));
    proof.transpose_array[static_cast<size_t>(i)] = (b != 0);
  }
  in.read(reinterpret_cast<char *>(proof.gl_left_array.data()),
          n * sizeof(uint16_t));
  in.read(reinterpret_cast<char *>(proof.gl_right_array.data()),
          n * sizeof(uint16_t));
  proof.Check();
  return proof;
}

void BacktrackingProof::Append(const BacktrackingProof &other) {
  dfs_restrictions_size_array.insert(dfs_restrictions_size_array.end(),
                                     other.dfs_restrictions_size_array.begin(),
                                     other.dfs_restrictions_size_array.end());
  mask_array.insert(mask_array.end(), other.mask_array.begin(),
                    other.mask_array.end());
  transpose_array.insert(transpose_array.end(), other.transpose_array.begin(),
                         other.transpose_array.end());
  gl_left_array.insert(gl_left_array.end(), other.gl_left_array.begin(),
                       other.gl_left_array.end());
  gl_right_array.insert(gl_right_array.end(), other.gl_right_array.begin(),
                        other.gl_right_array.end());
}
