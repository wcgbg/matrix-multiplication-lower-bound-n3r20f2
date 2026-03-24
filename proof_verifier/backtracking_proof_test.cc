#include "proof_verifier/backtracking_proof.h"

#include <cstdlib>
#include <string>

#include <gtest/gtest.h>

TEST(BacktrackingProofTest, SaveLoadRoundTripEmpty) {
  const char *tmp = std::getenv("TEST_TMPDIR");
  ASSERT_NE(tmp, nullptr);
  const std::string path = std::string(tmp) + "/backtracking_proof_empty.gz";

  BacktrackingProof proof;
  proof.Check();
  EXPECT_EQ(proof.Size(), 0u);
  proof.Save(path);
  BacktrackingProof loaded = BacktrackingProof::Load(path);
  EXPECT_EQ(loaded.Size(), 0u);
  EXPECT_TRUE(loaded.dfs_restrictions_size_array.empty());
  EXPECT_TRUE(loaded.mask_array.empty());
  EXPECT_TRUE(loaded.transpose_array.empty());
  EXPECT_TRUE(loaded.gl_left_array.empty());
  EXPECT_TRUE(loaded.gl_right_array.empty());
}

TEST(BacktrackingProofTest, SaveLoadRoundTripNonEmpty) {
  const char *tmp = std::getenv("TEST_TMPDIR");
  ASSERT_NE(tmp, nullptr);
  const std::string path = std::string(tmp) + "/backtracking_proof_nonempty.gz";

  BacktrackingProof proof;
  for (int i = 0; i < 100; i++) {
    proof.Append(i, i * 2u, i % 3 == 0, i * 4, i * 5);
  }
  proof.Check();
  EXPECT_EQ(proof.Size(), 100u);
  proof.Save(path);

  BacktrackingProof loaded = BacktrackingProof::Load(path);
  ASSERT_EQ(loaded.Size(), proof.Size());
  for (int i = 0; i < proof.Size(); i++) {
    EXPECT_EQ(loaded.dfs_restrictions_size_array[i],
              proof.dfs_restrictions_size_array[i]);
    EXPECT_EQ(loaded.mask_array[i], proof.mask_array[i]);
    EXPECT_EQ(loaded.transpose_array[i], proof.transpose_array[i]);
    EXPECT_EQ(loaded.gl_left_array[i], proof.gl_left_array[i]);
    EXPECT_EQ(loaded.gl_right_array[i], proof.gl_right_array[i]);
  }
}

TEST(BacktrackingProofTest, AppendOther) {
  BacktrackingProof a;
  a.Append(1, 100u, false, 1, 2);
  BacktrackingProof b;
  b.Append(2, 200u, true, 3, 4);
  b.Append(3, 300u, false, 5, 6);
  a.Append(b);
  EXPECT_EQ(a.Size(), 3u);
  EXPECT_EQ(a.dfs_restrictions_size_array[0], 1);
  EXPECT_EQ(a.dfs_restrictions_size_array[1], 2);
  EXPECT_EQ(a.dfs_restrictions_size_array[2], 3);
  EXPECT_EQ(a.mask_array[0], 100u);
  EXPECT_EQ(a.mask_array[1], 200u);
  EXPECT_EQ(a.mask_array[2], 300u);
  EXPECT_FALSE(a.transpose_array[0]);
  EXPECT_TRUE(a.transpose_array[1]);
  EXPECT_FALSE(a.transpose_array[2]);
  EXPECT_EQ(a.gl_left_array[0], 1);
  EXPECT_EQ(a.gl_left_array[1], 3);
  EXPECT_EQ(a.gl_left_array[2], 5);
  EXPECT_EQ(a.gl_right_array[0], 2);
  EXPECT_EQ(a.gl_right_array[1], 4);
  EXPECT_EQ(a.gl_right_array[2], 6);
}

TEST(BacktrackingProofTest, GetBacktrackingProofRootDir) {
  EXPECT_EQ(GetBacktrackingProofRootDir("rmms_n223.pb.txt"),
            "rmms_n223_bt_proof");
  EXPECT_EQ(GetBacktrackingProofRootDir("proof/rmms_n344.pb"),
            "proof/rmms_n344_bt_proof");
}

TEST(BacktrackingProofTest, GetBacktrackingProofPath) {
  EXPECT_EQ(GetBacktrackingProofPath("", 1234), "");

  const std::string root = "proof/rmms_n344_bt_proof";
  const std::string path = GetBacktrackingProofPath(root, 12345, false);
  EXPECT_EQ(path, "proof/rmms_n344_bt_proof/012/345.btp");
}
