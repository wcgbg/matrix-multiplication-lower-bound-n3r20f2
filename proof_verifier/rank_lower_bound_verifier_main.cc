#include <string>

#include <ng-log/logging.h>

#include "proof_verifier/backtracking_proof.h"
#include "proof_verifier/dimension.h"
#include "proof_verifier/proto_io.h"
#include "proof_verifier/rank_lower_bound_verifier.h"
#include "proof_verifier/restricted_mm.pb.h"

int main(int argc, char **argv) {
  FLAGS_alsologtostderr = true;
  FLAGS_log_dir = "/tmp";
  nglog::InitializeLogging(argv[0]);

  if (argc != 2) {
    LOG(FATAL) << "Usage: " << argv[0] << " <proto_path>";
  }
  std::string path = argv[1];

  pb::RestrictedMMCollection collection =
      ReadProtoFromFile<pb::RestrictedMMCollection>(path);
  std::string bt_proof_root_dir = GetBacktrackingProofRootDir(path);
  VerifyRankLowerBound<kN0, kN1, kN2>(collection, bt_proof_root_dir);

  return 0;
}
