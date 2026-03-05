#!/bin/bash

set -e

write_dimention_h() {
    local basename="${1##*/}"   # e.g. rmms_n223
    local n_part="${basename#rmms_n}"   # e.g. 223
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

for basename in testdata/rank_lower_bound_computer/rmms_n{123,132,213,231,312,321,222,223,232,322,233,323,332,333}; do
    echo
    echo "Processing ${basename}..."

    write_dimention_h ${basename}

    bazel build -c opt :restrictions_enumerator_main
    bazel-bin/restrictions_enumerator_main --output_path=${basename}.pb.txt
    
    bazel build -c opt :rank_lower_bound_computer_main
    output_path=${basename}_basic_degenerate_backtracking.pb.txt
    extra_args=""
    if [[ ${basename} == *332 || ${basename} == *233 || ${basename} == *323 || ${basename} == *333 ]]; then
        extra_args="--backtracking_step_limit=100000"
    fi
    bazel-bin/rank_lower_bound_computer_main ${basename}.pb.txt --output_path=${output_path} ${extra_args}
done
