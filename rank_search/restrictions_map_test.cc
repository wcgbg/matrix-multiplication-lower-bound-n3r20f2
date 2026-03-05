#include "restrictions_map.h"

#include <gtest/gtest.h>

#include "proof_verifier/math_utils.h"
#include "proof_verifier/restrictions.h"
#include "proof_verifier/static_matrix.h"

template <int n> StaticMatrix<n> RandomFullRankMatrix(std::mt19937_64 *gen) {
  while (true) {
    StaticMatrix<n> m = StaticMatrix<n>::Random(gen);
    if (m.Rank() == n) {
      return m;
    }
  }
}

template <int n0, int n1, int n2> void TestSetGet(std::mt19937_64 *gen) {
  RestrictionsMap<n0, n1, n2> map;
  constexpr int kNumTrials = 1000;
  for (int trial = 0; trial < kNumTrials; ++trial) {
    map.Clear();
    Restrictions<n0, n1> restrictions;
    for (int i = 0; i < n0 * n1; ++i) {
      restrictions.push_back(StaticMatrix<n0, n1>::Random(gen).Data());
    }
    int rank = GaussJordanElimination(n0 * n1, &restrictions);
    restrictions.erase(restrictions.begin(), restrictions.end() - rank);
    map.Set(restrictions, 42);

    StaticMatrix<n0> gl_left = RandomFullRankMatrix<n0>(gen);
    StaticMatrix<n1> gl_right = RandomFullRankMatrix<n1>(gen);
    bool transpose = false;
    if (n0 == n1 && n1 == n2) {
      transpose = (*gen)() % 2 == 0;
    }
    Restrictions<n0, n1> transformed = TransformRestrictions<n0, n1, n2>(
        restrictions, transpose, gl_left, gl_right);
    int get_rank = map.Get(transformed, &transpose, &gl_left, &gl_right);
    EXPECT_EQ(get_rank, 42);

    Restrictions<n0, n1> reconstructed = TransformRestrictions<n0, n1, n2>(
        transformed, gl_left, gl_right, transpose);
    EXPECT_EQ((RestrictionsToString<n0, n1>(restrictions)),
              (RestrictionsToString<n0, n1>(reconstructed)));
  }
}

TEST(RestrictionsMapTest, SetGetTransposeGlLeftGlRightInv) {
  std::mt19937_64 gen;
  TestSetGet<2, 2, 2>(&gen);
  TestSetGet<2, 3, 2>(&gen);
  TestSetGet<2, 2, 3>(&gen);
  TestSetGet<3, 3, 3>(&gen);
}
