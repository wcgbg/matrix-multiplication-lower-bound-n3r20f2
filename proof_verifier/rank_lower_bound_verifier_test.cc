#include "proof_verifier/rank_lower_bound_verifier.h"

#include <cstdlib>
#include <string>
#include <utility>

#include <gtest/gtest.h>

#include "proof_verifier/backtracking_proof.h"
#include "proof_verifier/proto_io.h"
#include "proof_verifier/restricted_mm.pb.h"

namespace {

std::pair<pb::RestrictedMMCollection, std::string> LoadRmmsN222Fixture() {
  const std::string pb_path = "proof_cert/rmms_n222.pb.txt";
  return {ReadProtoFromFile<pb::RestrictedMMCollection>(pb_path),
          GetBacktrackingProofRootDir(pb_path)};
}

} // namespace

TEST(RankLowerBoundVerifierTest, VerifiesRmmsN222) {
  auto [collection, bt_proof_root_dir] = LoadRmmsN222Fixture();

  VerifyRankLowerBound<2, 2, 2>(collection, bt_proof_root_dir);

  ASSERT_GT(collection.restricted_mm_size(), 0);
  const auto &last_rmm =
      collection.restricted_mm(collection.restricted_mm_size() - 1);
  EXPECT_GE(last_rmm.rank_lower_bound(), 7);
}

TEST(RankLowerBoundVerifierDeathTest, DetectsInflatedRankLowerBound) {
  auto [collection, bt_proof_root_dir] = LoadRmmsN222Fixture();
  ASSERT_GT(collection.restricted_mm_size(), 0);
  auto *last =
      collection.mutable_restricted_mm(collection.restricted_mm_size() - 1);
  last->set_rank_lower_bound(last->rank_lower_bound() + 1);

  EXPECT_DEATH((VerifyRankLowerBound<2, 2, 2>(collection, bt_proof_root_dir)),
               "");
}
