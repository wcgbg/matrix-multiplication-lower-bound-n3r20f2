#include "restrictions_enumerator.h"

#include <string>

#include <gtest/gtest.h>
#include <ng-log/logging.h>

#include "proof_verifier/proto_io.h"

template <int n0, int n1, int n2>
void TestRestrictionEnumerator(bool fill_verbose_fields) {
  RestrictionEnumerator<n0, n1, n2> restriction_enumerator;
  pb::RestrictedMMCollection collection =
      restriction_enumerator.Search(fill_verbose_fields);

  std::string golden_path =
      std::format("rank_search/testdata/rmms_n{}{}{}{}.pb.txt", n0, n1, n2,
                  fill_verbose_fields ? "_verbose" : "");
  pb::RestrictedMMCollection expected =
      ReadProtoFromFile<pb::RestrictedMMCollection>(golden_path);
  EXPECT_EQ(collection.SerializeAsString(), expected.SerializeAsString())
      << "n0=" << n0 << " n1=" << n1 << " n2=" << n2
      << " fill_verbose_fields=" << fill_verbose_fields;
}

TEST(RestrictionEnumeratorTest, All) {
  for (bool fill_verbose_fields : {false, true}) {
    TestRestrictionEnumerator<2, 2, 2>(fill_verbose_fields);
    TestRestrictionEnumerator<2, 2, 3>(fill_verbose_fields);
    TestRestrictionEnumerator<1, 2, 3>(fill_verbose_fields);
    TestRestrictionEnumerator<1, 3, 2>(fill_verbose_fields);
    TestRestrictionEnumerator<2, 1, 3>(fill_verbose_fields);
    TestRestrictionEnumerator<2, 3, 1>(fill_verbose_fields);
    TestRestrictionEnumerator<3, 1, 2>(fill_verbose_fields);
    TestRestrictionEnumerator<3, 2, 1>(fill_verbose_fields);
  }
}
