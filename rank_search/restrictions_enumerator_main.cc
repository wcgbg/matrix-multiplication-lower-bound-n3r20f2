#include <format>
#include <string>

#include <gflags/gflags.h>
#include <ng-log/logging.h>

#include "proof_verifier/dimension.h"
#include "proof_verifier/proto_io.h"
#include "proof_verifier/restricted_mm.pb.h"
#include "restrictions_enumerator.h"
#include "restrictions_enumerator_slow.h"

DEFINE_bool(slow, false, "Run the simple but slow version");
DEFINE_string(output_path, "", "Output path");

int main(int argc, char **argv) {
  FLAGS_alsologtostderr = true;
  FLAGS_log_dir = "/tmp";
  google::ParseCommandLineFlags(&argc, &argv, true);
  nglog::InitializeLogging(argv[0]);

  pb::RestrictedMMCollection collection;
  if (FLAGS_slow) {
    RestrictionEnumeratorSlow<kN0, kN1, kN2> restriction_enumerator;
    collection = restriction_enumerator.Search();
  } else {
    RestrictionEnumerator<kN0, kN1, kN2> restriction_enumerator;
    collection = restriction_enumerator.Search();
  }

  std::string output_path = FLAGS_output_path;
  if (output_path.empty()) {
    output_path = std::format("rmms_n{}{}{}.pb.txt", kN0, kN1, kN2);
  }
  WriteProtoToFile(collection, output_path);

  return 0;
}
