#include <limits>
#include <string>

#include <gflags/gflags.h>
#include <ng-log/logging.h>
#include <tbb/blocked_range.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>

#include "proof_verifier/dimension.h"
#include "proof_verifier/proto_io.h"
#include "proof_verifier/restricted_mm.pb.h"
#include "rank_lower_bound_computer.h"
#include "restrictions_map.h"

DEFINE_string(output_path, "", "Output path");
DEFINE_bool(ignore_rank_lower_bound, false, "Ignore rank lower bound");
DEFINE_uint64(backtracking_step_limit, std::numeric_limits<uint64_t>::max(),
              "Max number of backtracking search steps (all threads)");
DEFINE_int32(rank_lower_bound_min, 0, "Rank lower bound min (0 to n0*n1*n2)");
DEFINE_int32(rank_lower_bound_max, std::numeric_limits<int32_t>::max(),
             "Rank lower bound max (0 to n0*n1*n2)");
DEFINE_int32(restriction_size_min, 0, "Restriction size min (0 to n0*n1)");
DEFINE_int32(restriction_size_max, std::numeric_limits<int32_t>::max(),
             "Restriction size max (0 to n0*n1)");

int main(int argc, char **argv) {
  FLAGS_alsologtostderr = true;
  FLAGS_log_dir = "/tmp";
  google::ParseCommandLineFlags(&argc, &argv, true);
  nglog::InitializeLogging(argv[0]);

  if (argc != 2 && argc != 3) {
    LOG(FATAL) << "Usage: " << argv[0] << " <proto_path>";
  }
  std::string path = argv[1];

  // For debugging.
  // oneapi::tbb::global_control global_limit(
  //     oneapi::tbb::global_control::max_allowed_parallelism, 1);

  pb::RestrictedMMCollection collection =
      ReadProtoFromFile<pb::RestrictedMMCollection>(path);
  CHECK_GT(collection.restricted_mm_size(), 0);

  RestrictionsMap<kN0, kN1, kN2> restrictions_to_rank_lower_bound;

  if (FLAGS_ignore_rank_lower_bound) {
    for (pb::RestrictedMM &rmm : *collection.mutable_restricted_mm()) {
      rmm.clear_rank_lower_bound();
      rmm.clear_rank_lower_bound_proof();
    }
  }

  LOG(INFO) << "Building restrictions map...";
  BuildRestrictionsMap<kN0, kN1, kN2>(collection, FLAGS_rank_lower_bound_max,
                                      &restrictions_to_rank_lower_bound);

  std::string output_path = FLAGS_output_path;
  if (output_path.empty()) {
    if (path.ends_with(".pb.txt")) {
      std::string prefix = path.substr(0, path.size() - 7);
      output_path = prefix + "_updated.pb.txt";
    } else if (path.ends_with(".pb")) {
      std::string prefix = path.substr(0, path.size() - 3);
      output_path = prefix + "_updated.pb";
    } else {
      LOG(FATAL) << "Unsupported file extension: " << path;
    }
  }

  ProcessRestrictions<kN0, kN1>(
      {
          .basic_method = true,
          .degenerate_method = true,
          .backtracking_step_limit = FLAGS_backtracking_step_limit,
          .rank_lower_bound_min = FLAGS_rank_lower_bound_min,
          .rank_lower_bound_max = FLAGS_rank_lower_bound_max,
          .restriction_size_min = FLAGS_restriction_size_min,
          .restriction_size_max = FLAGS_restriction_size_max,
          .bt_proof_root_dir = GetBacktrackingProofRootDir(output_path),
      },
      output_path, &collection, &restrictions_to_rank_lower_bound);

  LOG(INFO) << "Done. Destructing...";
  return 0;
}
