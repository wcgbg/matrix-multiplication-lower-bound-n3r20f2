#include "rank_lower_bound_computer.h"

#include <string>

#include <boost/algorithm/string.hpp>
#include <gtest/gtest.h>
#include <ng-log/logging.h>
#include <tbb/global_control.h>

#include "proof_verifier/proto_io.h"
#include "restrictions_map.h"

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
  ASSERT_EQ(collection.n0(), expected_collection.n0());
  ASSERT_EQ(collection.n1(), expected_collection.n1());
  ASSERT_EQ(collection.n2(), expected_collection.n2());
  ASSERT_EQ(collection.p(), expected_collection.p());
  ASSERT_EQ(collection.restricted_mm_size(),
            expected_collection.restricted_mm_size());
  for (int i = 0; i < collection.restricted_mm_size(); ++i) {
    const pb::RestrictedMM &rmm = collection.restricted_mm(i);
    const pb::RestrictedMM &expected_rmm = expected_collection.restricted_mm(i);
    EXPECT_EQ(rmm.index(), expected_rmm.index());
    EXPECT_EQ(rmm.compact_restrictions(), expected_rmm.compact_restrictions());
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
