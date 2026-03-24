#!/bin/bash
set -e

write_dimention_h() {
    local basename="${1##*/}"   # e.g. rmms_n223.pb.txt
    local n_part="${basename#rmms_n}"   # e.g. 223.pb.txt
    local n0="${n_part:0:1}"
    local n1="${n_part:1:1}"
    local n2="${n_part:2:1}"
    cat > proof_verifier/dimension.h << EOF
#pragma once

constexpr int kN0 = $n0;
constexpr int kN1 = $n1;
constexpr int kN2 = $n2;
EOF
}

if [[ ! -d proof ]]; then
    echo "Please run this script from the root of the repository."
    exit 1
fi

if [[ $# -ne 1 && $# -ne 2 ]]; then
    echo "Usage: $0 [--use-gpu] proof/rmms_nXYZ.pb.txt"
    exit 1
fi

if [[ $1 == "--use-gpu" || $1 == "--use_gpu" ]]; then
    use_gpu=1
    proof_file=$2
else
    use_gpu=0
    proof_file=$1
fi

if [[ ! -f ${proof_file} ]]; then
    echo "File ${proof_file} does not exist."
    exit 1
fi

if [[ ${proof_file} == *".gz" ]]; then
    echo "File ${proof_file} is a gzipped file. Please uncompress it first."
    exit 1
fi

echo "Verifying ${proof_file}..."

write_dimention_h ${proof_file}

if [[ ${use_gpu} == 1 ]]; then
    bazel build -c opt --define=use_gpu=1 //proof_verifier:rank_lower_bound_verifier_main
else
    bazel build -c opt //proof_verifier:rank_lower_bound_verifier_main
fi

bazel-bin/proof_verifier/rank_lower_bound_verifier_main ${proof_file}
