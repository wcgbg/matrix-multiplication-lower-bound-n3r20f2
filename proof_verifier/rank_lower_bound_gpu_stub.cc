#include "proof_verifier/rank_lower_bound_gpu.h"

// Single translation unit provides the symbol when not building with CUDA.
bool IsCudaAvailable() { return false; }

template <int n0, int n1, int n2>
std::optional<int> RankLowerBoundForcedProductALoopCuda(
    const RankLowerBoundForcedProductALoopParams<n0, n1, n2> &params) {
  LOG(FATAL) << "RankLowerBoundForcedProductALoopCuda stub";
  return std::nullopt;
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
