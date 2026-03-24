// CUDA implementation of the rank lower bound loop.

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <optional>
#include <queue>

#include <cuda_runtime.h>

#include "proof_verifier/rank_lower_bound_gpu.h"

namespace {

// F2 (GF(2)) matrix rank via Gaussian elimination (XOR only).
// Matrix is row-major, modified in place. Returns rank.
__device__ int F2MatrixRank(int rows, int cols, uint8_t *mat) {
  int rank = 0;
  int pivot_col = 0;

  for (int row = 0; row < rows && pivot_col < cols; ++row) {
    int pivot_row = -1;
    for (int r = row; r < rows; ++r) {
      if (mat[r * cols + pivot_col] != 0) {
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
      for (int c = 0; c < cols; ++c) {
        uint8_t t = mat[row * cols + c];
        mat[row * cols + c] = mat[pivot_row * cols + c];
        mat[pivot_row * cols + c] = t;
      }
    }

    for (int r = 0; r < rows; ++r) {
      if (r != row && mat[r * cols + pivot_col] != 0) {
        for (int c = 0; c < cols; ++c) {
          mat[r * cols + c] ^= mat[row * cols + c];
        }
      }
    }

    ++rank;
    ++pivot_col;
  }
  return rank;
}

// Flatten tensor to AxBC matrix (n01 x n12*n20), write into out (row-major).
template <int n0, int n1, int n2>
__device__ void FlattenAxBC(const uint8_t *tensor, uint8_t *out) {
  constexpr int n01 = n0 * n1;
  constexpr int n12 = n1 * n2;
  constexpr int n20 = n2 * n0;
  for (int abc = 0; abc < n01 * n12 * n20; ++abc) {
    out[abc] = tensor[abc];
  }
}

// Flatten tensor to BxAC matrix (n12 x n01*n20).
template <int n0, int n1, int n2>
__device__ void FlattenBxAC(const uint8_t *tensor, uint8_t *out) {
  constexpr int n01 = n0 * n1;
  constexpr int n12 = n1 * n2;
  constexpr int n20 = n2 * n0;
  for (int b = 0; b < n12; ++b) {
    for (int a = 0; a < n01; ++a) {
      for (int c = 0; c < n20; ++c) {
        out[b * (n01 * n20) + a * n20 + c] =
            tensor[a * (n12 * n20) + b * n20 + c];
      }
    }
  }
}

// Flatten tensor to ABxC matrix (n01*n12 x n20).
template <int n0, int n1, int n2>
__device__ void FlattenABxC(const uint8_t *tensor, uint8_t *out) {
  constexpr int n01 = n0 * n1;
  constexpr int n12 = n1 * n2;
  constexpr int n20 = n2 * n0;
  for (int abc = 0; abc < n01 * n12 * n20; ++abc) {
    out[abc] = tensor[abc];
  }
}

// Rank lower bound for one tensor on device. projection_type starts at 1.
// Uses shared logic: try BxAC first, then ABxC, then AxBC; return max rank
// with early exit if >= target_rank. Workspace is thread-local (caller
// provides).
template <int n0, int n1, int n2>
__device__ int RankLowerBoundFlattenMatrixDevice(const uint8_t *tensor,
                                                 int target_rank,
                                                 uint8_t *workspace) {
  constexpr int n01 = n0 * n1;
  constexpr int n12 = n1 * n2;
  constexpr int n20 = n2 * n0;

  // projection_type == 1: try BxAC first
  FlattenBxAC<n0, n1, n2>(tensor, workspace);
  int rank1 = F2MatrixRank(n12, n01 * n20, workspace);
  if (rank1 >= target_rank)
    return rank1;

  FlattenABxC<n0, n1, n2>(tensor, workspace);
  int rank2 = F2MatrixRank(n01 * n12, n20, workspace);
  if (rank2 >= target_rank)
    return rank2;

  FlattenAxBC<n0, n1, n2>(tensor, workspace);
  int rank0 = F2MatrixRank(n01, n12 * n20, workspace);
  if (rank0 >= target_rank)
    return rank0;

  return std::max(rank0, std::max(rank1, rank2));
}

// Device params (pointers and scalars). r2p and r1_bc
// are copied to device buffers; we pass pointers.
template <int n0, int n1, int n2> struct DeviceParams {
  const uint8_t *d_r2p = nullptr;
  const uint8_t *d_r1_bc = nullptr;
  int r1_bc_rows = 0;
  int r2p_size0 = 0;
  int bit_width = 0;
  int known_lower_bound = 0;
};

template <int n0, int n1, int n2>
__global__ void
RankLowerBoundForcedProductALoopKernel(const DeviceParams<n0, n1, n2> params,
                                       uint64_t chunk_start, uint64_t chunk_end,
                                       int *d_min_result) {
  constexpr int n01 = n0 * n1;
  constexpr int n12 = n1 * n2;
  constexpr int n20 = n2 * n0;

  const uint64_t total_threads = static_cast<uint64_t>(blockDim.x) * gridDim.x;
  const uint64_t thread_idx =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  int local_min = std::numeric_limits<int>::max();

  constexpr uint64_t prime = 73074167;
  const uint64_t mask = (uint64_t(1) << params.bit_width) - 1;

  for (uint64_t t = chunk_start + thread_idx; t < chunk_end;
       t += total_threads) {
    uint64_t binary = (t * prime) & mask;

    // Build tensor_t in local memory (r2p then XOR selected rows)
    uint8_t tensor_t[n01][n12][n20];
    // Copy r2p (layout: first r2p_size0 slices are used)
    for (int i = 0; i < params.r2p_size0; ++i) {
      for (int j = 0; j < n12; ++j) {
        for (int k = 0; k < n20; ++k) {
          tensor_t[i][j][k] = params.d_r2p[i * (n12 * n20) + j * n20 + k];
        }
      }
    }
    for (int i = params.r2p_size0; i < n01; ++i) {
      for (int j = 0; j < n12; ++j) {
        for (int k = 0; k < n20; ++k) {
          tensor_t[i][j][k] = 0;
        }
      }
    }

    for (int bit_idx = 0; bit_idx < params.bit_width; ++bit_idx) {
      if (((binary >> bit_idx) & 1) == 0)
        continue;
      int r1_idx = bit_idx % params.r1_bc_rows;
      int i = bit_idx / params.r1_bc_rows;
      for (int j = 0; j < n12; ++j) {
        for (int k = 0; k < n20; ++k) {
          tensor_t[i][j][k] ^=
              params.d_r1_bc[r1_idx * (n12 * n20) + j * n20 + k];
        }
      }
    }

    uint8_t workspace[n01 * n12 * n20]; // thread-local
    int remaining = RankLowerBoundFlattenMatrixDevice<n0, n1, n2>(
        &tensor_t[0][0][0], local_min - params.r1_bc_rows, workspace);
    int candidate = params.r1_bc_rows + remaining;
    if (candidate < local_min)
      local_min = candidate;
  }

  // Block reduction then atomic min (or direct atomic per thread)
  __shared__ int block_min;
  if (threadIdx.x == 0)
    block_min = std::numeric_limits<int>::max();
  __syncthreads();
  atomicMin(&block_min, local_min);
  __syncthreads();
  if (threadIdx.x == 0) {
    atomicMin(d_min_result, block_min);
  }
}

// GPU pool: threads acquire a free device index, run, then release.
static std::mutex g_pool_mutex;
static std::condition_variable g_pool_cv;
static std::queue<int> g_free_devices;
static bool g_pool_initialized = false;

int AcquireGpu() {
  std::unique_lock<std::mutex> lock(g_pool_mutex);
  if (!g_pool_initialized) {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count <= 0) {
      return -1;
    }
    for (int i = 0; i < device_count; ++i) {
      g_free_devices.push(i);
    }
    g_pool_initialized = true;
  }
  while (g_free_devices.empty()) {
    g_pool_cv.wait(lock);
  }
  int device_id = g_free_devices.front();
  g_free_devices.pop();
  return device_id;
}

void ReleaseGpu(int device_id) {
  std::lock_guard<std::mutex> lock(g_pool_mutex);
  g_free_devices.push(device_id);
  g_pool_cv.notify_one();
}

struct GpuGuard {
  int device_id;
  ~GpuGuard() { ReleaseGpu(device_id); }
};

} // namespace

bool IsCudaAvailable() {
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess || device_count == 0) {
    return false;
  }
  return true;
}

template <int n0, int n1, int n2>
std::optional<int> RankLowerBoundForcedProductALoopCuda(
    const RankLowerBoundForcedProductALoopParams<n0, n1, n2> &params) {
  constexpr int n01 = n0 * n1;
  constexpr int n12 = n1 * n2;
  constexpr int n20 = n2 * n0;
  constexpr size_t tensor_size = n01 * n12 * n20;
  constexpr int r1_bc_cols = n12 * n20;

  if (params.r1_bc_data == nullptr || params.r1_bc_rows == 0) {
    printf("%s:%d\n", __FILE__, __LINE__);
    return std::nullopt;
  }

  int device_id = AcquireGpu();
  if (device_id < 0) {
    printf("%s:%d: AcquireGpu failed\n", __FILE__, __LINE__);
    return std::nullopt;
  }
  auto start_time = std::chrono::steady_clock::now();
  GpuGuard gpu_guard{device_id};
  cudaError_t err = cudaSetDevice(device_id);
  if (err != cudaSuccess) {
    printf("%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
    return std::nullopt;
  }

  uint8_t *d_r2p = nullptr;
  uint8_t *d_r1_bc = nullptr;
  int *d_min_result = nullptr;
  err = cudaMalloc(&d_r2p, tensor_size);
  if (err != cudaSuccess) {
    printf("%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
    return std::nullopt;
  }
  err =
      cudaMalloc(&d_r1_bc, static_cast<size_t>(params.r1_bc_rows * r1_bc_cols));
  if (err != cudaSuccess) {
    printf("%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
    cudaFree(d_r2p);
    return std::nullopt;
  }
  err = cudaMalloc(&d_min_result, sizeof(int));
  if (err != cudaSuccess) {
    printf("%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
    cudaFree(d_r1_bc);
    cudaFree(d_r2p);
    return std::nullopt;
  }

  // Copy r2p (flat layout matching our kernel)
  const uint8_t *src_tensor =
      reinterpret_cast<const uint8_t *>(&params.r2p[0][0][0]);
  err = cudaMemcpy(d_r2p, src_tensor, tensor_size, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    printf("%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
    cudaFree(d_min_result);
    cudaFree(d_r1_bc);
    cudaFree(d_r2p);
    return std::nullopt;
  }

  err = cudaMemcpy(d_r1_bc, params.r1_bc_data,
                   static_cast<size_t>(params.r1_bc_rows * r1_bc_cols),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    printf("%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
    cudaFree(d_min_result);
    cudaFree(d_r1_bc);
    cudaFree(d_r2p);
    return std::nullopt;
  }

  DeviceParams<n0, n1, n2> d_params;
  d_params.d_r2p = d_r2p;
  d_params.d_r1_bc = d_r1_bc;
  d_params.r1_bc_rows = params.r1_bc_rows;
  d_params.r2p_size0 = params.r2p_size0;
  d_params.bit_width = params.bit_width;
  d_params.known_lower_bound = params.known_lower_bound;

  const uint64_t num_iterations = uint64_t(1) << params.bit_width;

  constexpr uint64_t kChunkSize = 1ULL << 20; // 1M iterations per chunk
  int global_min = std::numeric_limits<int>::max();
  bool early_break = false;

  int h_initial_min = std::numeric_limits<int>::max();
  err = cudaMemcpy(d_min_result, &h_initial_min, sizeof(int),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    printf("%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
    cudaFree(d_min_result);
    cudaFree(d_r1_bc);
    cudaFree(d_r2p);
    return std::nullopt;
  }

  for (uint64_t chunk_start = 0; chunk_start < num_iterations;
       chunk_start += kChunkSize) {
    if (chunk_start > 0 && chunk_start % (kChunkSize * 256) == 0) {
      printf("%s:%d] [device_id=%d] %lu / %lu = %.2f%%\n", __FILE__, __LINE__,
             device_id, chunk_start, num_iterations,
             static_cast<double>(chunk_start) / num_iterations * 100.0);
    }
    uint64_t chunk_end = std::min(chunk_start + kChunkSize, num_iterations);

    err = cudaMemcpy(d_min_result, &h_initial_min, sizeof(int),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      printf("%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
      break;
    }

    RankLowerBoundForcedProductALoopKernel<n0, n1, n2>
        <<<256, 256>>>(d_params, chunk_start, chunk_end, d_min_result);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
      break;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      printf("%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
      break;
    }

    int chunk_min = std::numeric_limits<int>::max();
    err = cudaMemcpy(&chunk_min, d_min_result, sizeof(int),
                     cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      printf("%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
      break;
    }

    if (chunk_min < global_min)
      global_min = chunk_min;
    if (global_min <= params.known_lower_bound) {
      early_break = true;
      break;
    }
  }

  cudaFree(d_min_result);
  cudaFree(d_r1_bc);
  cudaFree(d_r2p);

  if (err != cudaSuccess) {
    return std::nullopt;
  }

  int ret = early_break ? -1 : global_min;

  auto end_time = std::chrono::steady_clock::now();
  auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                         end_time - start_time)
                         .count();
  if (duration_ms > 1000) {
    printf("%s:%d] [device_id=%d] rank=%d duration=%.2f\n", __FILE__, __LINE__,
           device_id, ret, duration_ms / 1000.0);
  }

  return ret;
}

// Explicit instantiations for common dimensions.
template std::optional<int> RankLowerBoundForcedProductALoopCuda<2, 2, 2>(
    const RankLowerBoundForcedProductALoopParams<2, 2, 2> &);

template std::optional<int> RankLowerBoundForcedProductALoopCuda<3, 3, 3>(
    const RankLowerBoundForcedProductALoopParams<3, 3, 3> &);

template std::optional<int> RankLowerBoundForcedProductALoopCuda<2, 4, 3>(
    const RankLowerBoundForcedProductALoopParams<2, 4, 3> &);
template std::optional<int> RankLowerBoundForcedProductALoopCuda<3, 2, 4>(
    const RankLowerBoundForcedProductALoopParams<3, 2, 4> &);
template std::optional<int> RankLowerBoundForcedProductALoopCuda<4, 3, 2>(
    const RankLowerBoundForcedProductALoopParams<4, 3, 2> &);

template std::optional<int> RankLowerBoundForcedProductALoopCuda<2, 3, 4>(
    const RankLowerBoundForcedProductALoopParams<2, 3, 4> &);
template std::optional<int> RankLowerBoundForcedProductALoopCuda<3, 4, 2>(
    const RankLowerBoundForcedProductALoopParams<3, 4, 2> &);
template std::optional<int> RankLowerBoundForcedProductALoopCuda<4, 2, 3>(
    const RankLowerBoundForcedProductALoopParams<4, 2, 3> &);

template std::optional<int> RankLowerBoundForcedProductALoopCuda<1, 2, 3>(
    const RankLowerBoundForcedProductALoopParams<1, 2, 3> &);
template std::optional<int> RankLowerBoundForcedProductALoopCuda<3, 1, 2>(
    const RankLowerBoundForcedProductALoopParams<3, 1, 2> &);
template std::optional<int> RankLowerBoundForcedProductALoopCuda<2, 3, 1>(
    const RankLowerBoundForcedProductALoopParams<2, 3, 1> &);

template std::optional<int> RankLowerBoundForcedProductALoopCuda<1, 3, 2>(
    const RankLowerBoundForcedProductALoopParams<1, 3, 2> &);
template std::optional<int> RankLowerBoundForcedProductALoopCuda<2, 1, 3>(
    const RankLowerBoundForcedProductALoopParams<2, 1, 3> &);
template std::optional<int> RankLowerBoundForcedProductALoopCuda<3, 2, 1>(
    const RankLowerBoundForcedProductALoopParams<3, 2, 1> &);

template std::optional<int> RankLowerBoundForcedProductALoopCuda<2, 2, 3>(
    const RankLowerBoundForcedProductALoopParams<2, 2, 3> &);
template std::optional<int> RankLowerBoundForcedProductALoopCuda<3, 2, 2>(
    const RankLowerBoundForcedProductALoopParams<3, 2, 2> &);
template std::optional<int> RankLowerBoundForcedProductALoopCuda<2, 3, 2>(
    const RankLowerBoundForcedProductALoopParams<2, 3, 2> &);

template std::optional<int> RankLowerBoundForcedProductALoopCuda<2, 3, 3>(
    const RankLowerBoundForcedProductALoopParams<2, 3, 3> &);
template std::optional<int> RankLowerBoundForcedProductALoopCuda<3, 2, 3>(
    const RankLowerBoundForcedProductALoopParams<3, 2, 3> &);
template std::optional<int> RankLowerBoundForcedProductALoopCuda<3, 3, 2>(
    const RankLowerBoundForcedProductALoopParams<3, 3, 2> &);

template std::optional<int> RankLowerBoundForcedProductALoopCuda<3, 3, 4>(
    const RankLowerBoundForcedProductALoopParams<3, 3, 4> &);
template std::optional<int> RankLowerBoundForcedProductALoopCuda<4, 3, 3>(
    const RankLowerBoundForcedProductALoopParams<4, 3, 3> &);
template std::optional<int> RankLowerBoundForcedProductALoopCuda<3, 4, 3>(
    const RankLowerBoundForcedProductALoopParams<3, 4, 3> &);

template std::optional<int> RankLowerBoundForcedProductALoopCuda<3, 4, 4>(
    const RankLowerBoundForcedProductALoopParams<3, 4, 4> &);
template std::optional<int> RankLowerBoundForcedProductALoopCuda<4, 3, 4>(
    const RankLowerBoundForcedProductALoopParams<4, 3, 4> &);
template std::optional<int> RankLowerBoundForcedProductALoopCuda<4, 4, 3>(
    const RankLowerBoundForcedProductALoopParams<4, 4, 3> &);
