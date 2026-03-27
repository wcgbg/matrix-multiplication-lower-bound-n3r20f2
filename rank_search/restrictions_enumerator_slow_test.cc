#include "restrictions_enumerator_slow.h"

#include <string>

#include <gtest/gtest.h>
#include <ng-log/logging.h>

#include "proof_verifier/proto_io.h"

template <int n0, int n1, int n2> void TestRestrictionEnumeratorSlow() {
  RestrictionEnumeratorSlow<n0, n1, n2> restriction_enumerator;
  pb::RestrictedMMCollection collection = restriction_enumerator.Search();

  std::string golden_path =
      std::format("rank_search/testdata/rmms_n{}{}{}.pb.txt", n0, n1, n2);
  pb::RestrictedMMCollection expected =
      ReadProtoFromFile<pb::RestrictedMMCollection>(golden_path);
  EXPECT_EQ(collection.SerializeAsString(), expected.SerializeAsString())
      << "n0=" << n0 << " n1=" << n1 << " n2=" << n2;
}

TEST(RestrictionEnumeratorSlowTest, All) {
  TestRestrictionEnumeratorSlow<2, 2, 2>();

  TestRestrictionEnumeratorSlow<2, 2, 3>();

  TestRestrictionEnumeratorSlow<1, 2, 3>();
  TestRestrictionEnumeratorSlow<1, 3, 2>();
  TestRestrictionEnumeratorSlow<2, 1, 3>();
  TestRestrictionEnumeratorSlow<2, 3, 1>();
  TestRestrictionEnumeratorSlow<3, 1, 2>();
  TestRestrictionEnumeratorSlow<3, 2, 1>();
}
