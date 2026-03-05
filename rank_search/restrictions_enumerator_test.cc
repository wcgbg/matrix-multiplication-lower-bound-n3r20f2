#include "restrictions_enumerator.h"

#include <string>

#include <gtest/gtest.h>
#include <ng-log/logging.h>

#include "proof_verifier/proto_io.h"

template <int n0, int n1, int n2> void TestRestrictionEnumerator() {
  RestrictionEnumerator<n0, n1, n2> restriction_enumerator;
  pb::RestrictedMMCollection collection = restriction_enumerator.Search();

  std::string golden_path =
      std::format("rank_search/testdata/rmms_n{}{}{}.pb.txt", n0, n1, n2);
  pb::RestrictedMMCollection expected =
      ReadProtoFromFile<pb::RestrictedMMCollection>(golden_path);
  EXPECT_EQ(collection.SerializeAsString(), expected.SerializeAsString())
      << "n0=" << n0 << " n1=" << n1 << " n2=" << n2;
}

TEST(RestrictionEnumeratorTest, All) {
  TestRestrictionEnumerator<2, 2, 2>();

  TestRestrictionEnumerator<2, 2, 3>();

  TestRestrictionEnumerator<1, 2, 3>();
  TestRestrictionEnumerator<1, 3, 2>();
  TestRestrictionEnumerator<2, 1, 3>();
  TestRestrictionEnumerator<2, 3, 1>();
  TestRestrictionEnumerator<3, 1, 2>();
  TestRestrictionEnumerator<3, 2, 1>();
}
