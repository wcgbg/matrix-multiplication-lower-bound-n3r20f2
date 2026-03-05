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

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 proof/rmms_nXYZ.pb.txt"
    exit 1
fi

if [[ ! -f $1 ]]; then
    echo "File $1 does not exist."
    exit 1
fi

echo "Verifying $1..."

write_dimention_h $1

bazel build -c opt //proof_verifier:rank_lower_bound_verifier_main
bazel-bin/proof_verifier/rank_lower_bound_verifier_main $1
