#include <string>

#include <gflags/gflags.h>
#include <ng-log/logging.h>

#include "proof_verifier/dimension.h"
#include "proof_verifier/proto_io.h"
#include "proof_verifier/restricted_mm.pb.h"
#include "proof_verifier/restrictions.h"
#include "proof_verifier/tensor.h"
#include "proof_verifier/tensor_utils.h"

DEFINE_string(output_path, "",
              "Output path. Defaults to overwriting the input file.");

int main(int argc, char **argv) {
  FLAGS_alsologtostderr = true;
  FLAGS_log_dir = "/tmp";
  google::ParseCommandLineFlags(&argc, &argv, true);
  nglog::InitializeLogging(argv[0]);

  if (argc != 2) {
    LOG(FATAL) << "Usage: " << argv[0] << " <proto_path>";
  }
  const std::string input_path = argv[1];

  pb::RestrictedMMCollection collection =
      ReadProtoFromFile<pb::RestrictedMMCollection>(input_path);
  CHECK_EQ(collection.n0(), kN0);
  CHECK_EQ(collection.n1(), kN1);
  CHECK_EQ(collection.n2(), kN2);

  const Tensor<kN0, kN1, kN2> matrix_multiplication_tensor =
      MatrixMultiplicationTensor<kN0, kN1, kN2>();

  for (pb::RestrictedMM &rmm : *collection.mutable_restricted_mm()) {
    Restrictions<kN0, kN1> restrictions = RestrictionsFromProto<kN0, kN1>(rmm);
    rmm.set_restrictions_text(RestrictionsToString<kN0, kN1>(restrictions));
    rmm.set_tensor(TensorToSparseString<kN0, kN1, kN2>(
        ApplyRestrictionsToTensor<kN0, kN1, kN2>(
            restrictions, matrix_multiplication_tensor)));
  }

  const std::string output_path =
      FLAGS_output_path.empty() ? input_path : FLAGS_output_path;
  WriteProtoToFile(collection, output_path);

  return 0;
}
