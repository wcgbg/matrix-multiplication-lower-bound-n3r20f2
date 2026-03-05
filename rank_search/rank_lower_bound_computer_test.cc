#include "rank_lower_bound_computer.h"

#include <string>

#include <boost/algorithm/string.hpp>
#include <gtest/gtest.h>
#include <ng-log/logging.h>
#include <tbb/global_control.h>

#include "proof_verifier/proto_io.h"
#include "restrictions_map.h"

TEST(RankLowerBoundForcedProductATest, N2Rank2) {
  constexpr int n = 2;
  const std::string tensor_str = "0000,0000,0000,0000,\n"
                                 "0000,0000,0001,0000,\n"
                                 "0000,0000,0000,0000,\n"
                                 "0000,0000,0000,0001,\n";
  SquareTensor<n> tensor = DenseStringToTensor<n, n, n>(tensor_str);
  EXPECT_EQ((TensorToSparseString<n, n, n>(tensor)),
            "a01*b10*c11 + a11*b11*c11");

  int rank = RankLowerBoundForcedProductA<n, n, n>(tensor);

  EXPECT_EQ(rank, 2);
}

TEST(RankLowerBoundForcedProductATest, N2Rank6) {
  constexpr int n = 2;
  const std::string tensor_str = "a01*b00*c01 + a01*b01*c11 + a01*b10*c00 + "
                                 "a01*b11*c10 + a11*b10*c01 + a11*b11*c11";
  SquareTensor<n> tensor = SparseStringToTensor<n, n, n>(tensor_str);
  EXPECT_EQ((TensorToSparseString<n, n, n>(tensor)), tensor_str);
  EXPECT_EQ((RankLowerBoundForcedProductA<n, n, n>(tensor)), 0);
  tensor = CyclicTranspose<n, n, n>(tensor);
  EXPECT_EQ((RankLowerBoundForcedProductA<n, n, n>(tensor)), 6);
  tensor = CyclicTranspose<n, n, n>(tensor);
  EXPECT_EQ((RankLowerBoundForcedProductA<n, n, n>(tensor)), 6);
}

TEST(RankLowerBoundForcedProductATest, N3Rank9) {
  constexpr int n = 3;
  const std::string tensor_str =
      "a11*b00*c02 + a11*b01*c12 + a11*b02*c22 + a11*b10*c01 + a11*b11*c11 + "
      "a11*b12*c21 + a12*b10*c02 + a12*b11*c12 + a12*b12*c22 + a12*b20*c01 + "
      "a12*b21*c11 + a12*b22*c21";
  SquareTensor<n> tensor = SparseStringToTensor<n, n, n>(tensor_str);
  EXPECT_EQ((TensorToSparseString<n, n, n>(tensor)), tensor_str);
  tensor = CyclicTranspose<n, n, n>(tensor);
  EXPECT_EQ((RankLowerBoundForcedProductA<n, n, n>(tensor)), 9);
}

TEST(RankLowerBoundForcedProductATest, N3Rank15) {
  constexpr int n = 3;
  const std::string tensor_str =
      "a02*b20*c00 + a02*b21*c10 + a02*b22*c20 + a12*b20*c01 + a12*b21*c11 + "
      "a12*b22*c21 + a20*b00*c02 + a20*b01*c12 + a20*b02*c22 + a21*b10*c02 + "
      "a21*b11*c12 + a21*b12*c22 + a22*b20*c02 + a22*b21*c12 + a22*b22*c22";
  SquareTensor<n> tensor = SparseStringToTensor<n, n, n>(tensor_str);
  EXPECT_EQ((TensorToSparseString<n, n, n>(tensor)), tensor_str);

  EXPECT_EQ((RankLowerBoundForcedProductA<n, n, n>(tensor)),
            0); // r1_count == 0
  tensor = CyclicTranspose<n, n, n>(tensor);
  EXPECT_EQ((RankLowerBoundForcedProductA<n, n, n>(tensor)), 15);
  tensor = CyclicTranspose<n, n, n>(tensor);
  EXPECT_EQ((RankLowerBoundForcedProductA<n, n, n>(tensor)), 15);
}

template <int n0, int n1, int n2, const char *input_path_pattern,
          const char *expected_output_path_pattern>
void TestRankLowerBound(bool basic_method, bool degenerate_method,
                        uint64_t backtracking_step_limit) {
  std::string input_path = std::format(input_path_pattern, n0, n1, n2);
  std::string expected_output_path =
      std::format(expected_output_path_pattern, n0, n1, n2);
  LOG(INFO) << "basic_method=" << basic_method
            << " degenerate_method=" << degenerate_method
            << " backtracking_step_limit=" << backtracking_step_limit
            << " input_path=" << input_path
            << " expected_output_path=" << expected_output_path;

  pb::RestrictedMMCollection collection =
      ReadProtoFromFile<pb::RestrictedMMCollection>(input_path);
  int rank_lower_bound_min = 0;
  int rank_lower_bound_max = std::numeric_limits<int>::max();
  RestrictionsMap<n0, n1, n2> restrictions_to_rank_lower_bound;
  BuildRestrictionsMap<n0, n1, n2>(collection, rank_lower_bound_max,
                                   &restrictions_to_rank_lower_bound);
  bool has_update = ProcessRestrictions<n0, n1>(
      {
          .basic_method = basic_method,
          .degenerate_method = degenerate_method,
          .backtracking_step_limit = backtracking_step_limit,
          .rank_lower_bound_min = rank_lower_bound_min,
          .rank_lower_bound_max = rank_lower_bound_max,
      },
      "", &collection, &restrictions_to_rank_lower_bound);
  EXPECT_TRUE(has_update);
  pb::RestrictedMMCollection expected_collection =
      ReadProtoFromFile<pb::RestrictedMMCollection>(expected_output_path);

  ASSERT_EQ(collection.restricted_mm_size(),
            expected_collection.restricted_mm_size());
  for (int i = 0; i < collection.restricted_mm_size(); ++i) {
    const pb::RestrictedMM &rmm = collection.restricted_mm(i);
    const pb::RestrictedMM &expected_rmm = expected_collection.restricted_mm(i);
    EXPECT_EQ(rmm.index(), expected_rmm.index());
    EXPECT_EQ(rmm.n0(), expected_rmm.n0());
    EXPECT_EQ(rmm.n1(), expected_rmm.n1());
    EXPECT_EQ(rmm.n2(), expected_rmm.n2());
    EXPECT_EQ(rmm.p(), expected_rmm.p());
    EXPECT_EQ(rmm.tensor(), expected_rmm.tensor());
    EXPECT_EQ(rmm.rank_lower_bound(), expected_rmm.rank_lower_bound())
        << "rmm_index=" << rmm.index();
  }
}

namespace {
constexpr const char kPathPattern[] =
    "rank_search/testdata/rmms_n{}{}{}.pb.txt";
constexpr const char kBasicPathPattern[] =
    "rank_search/testdata/rmms_n{}{}{}_basic.pb.txt";
constexpr const char kDegeneratePathPattern[] =
    "rank_search/testdata/rmms_n{}{}{}_basic_degenerate.pb.txt";
constexpr const char kBacktrackingPathPattern[] =
    "rank_search/testdata/rmms_n{}{}{}_basic_degenerate_backtracking.pb.txt";
} // namespace

TEST(RankLowerBoundTest, StepByStep) {
  constexpr int n = 2;
  TestRankLowerBound<n, n, n, kPathPattern, kBasicPathPattern>(true, false, 0);
  TestRankLowerBound<n, n, n, kBasicPathPattern, kDegeneratePathPattern>(
      false, true, 0);
  TestRankLowerBound<n, n, n, kDegeneratePathPattern, kBacktrackingPathPattern>(
      false, true, std::numeric_limits<uint64_t>::max());
}

template <int n0, int n1, int n2> void TestRankLowerBoundAll() {
  uint64_t backtracking_step_limit =
      n0 * n1 * n2 <= 15 ? std::numeric_limits<uint64_t>::max() : 100000;
  TestRankLowerBound<n0, n1, n2, kPathPattern, kBacktrackingPathPattern>(
      true, true, backtracking_step_limit);
}

TEST(RankLowerBoundTest, All) {
  TestRankLowerBoundAll<1, 2, 3>();
  TestRankLowerBoundAll<1, 3, 2>();
  TestRankLowerBoundAll<2, 1, 3>();
  TestRankLowerBoundAll<2, 3, 1>();
  TestRankLowerBoundAll<3, 1, 2>();
  TestRankLowerBoundAll<3, 2, 1>();

  TestRankLowerBoundAll<2, 2, 3>();
  TestRankLowerBoundAll<2, 3, 2>();
  TestRankLowerBoundAll<3, 2, 2>();

  TestRankLowerBoundAll<2, 3, 3>();
  TestRankLowerBoundAll<3, 2, 3>();
  TestRankLowerBoundAll<3, 3, 2>();

  TestRankLowerBoundAll<3, 3, 3>();
}
