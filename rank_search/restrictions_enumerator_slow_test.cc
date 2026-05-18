#include "restrictions_enumerator_slow.h"

#include <string>

#include <gtest/gtest.h>
#include <ng-log/logging.h>

#include "proof_verifier/proto_io.h"

template <int n0, int n1, int n2>
void TestRestrictionEnumeratorSlow(bool fill_verbose_fields) {
  RestrictionEnumeratorSlow<n0, n1, n2> restriction_enumerator;
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

TEST(RestrictionEnumeratorSlowTest, All) {
  for (bool fill_verbose_fields : {false, true}) {
    TestRestrictionEnumeratorSlow<2, 2, 2>(fill_verbose_fields);
    TestRestrictionEnumeratorSlow<2, 2, 3>(fill_verbose_fields);
    TestRestrictionEnumeratorSlow<1, 2, 3>(fill_verbose_fields);
    TestRestrictionEnumeratorSlow<1, 3, 2>(fill_verbose_fields);
    TestRestrictionEnumeratorSlow<2, 1, 3>(fill_verbose_fields);
    TestRestrictionEnumeratorSlow<2, 3, 1>(fill_verbose_fields);
    TestRestrictionEnumeratorSlow<3, 1, 2>(fill_verbose_fields);
    TestRestrictionEnumeratorSlow<3, 2, 1>(fill_verbose_fields);
  }
}
