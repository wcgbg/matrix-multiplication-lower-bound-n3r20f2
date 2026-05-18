# Proof Verifier

This directory contains a standalone verifier for the computer-generated tensor rank lower bound proofs.

Proof certificates live in the top-level `proof_cert/` directory as text-format protobufs named `proof_cert/rmms_nXYZ.pb.txt` and binary files named `proof_cert/rmms_nXYZ_bt_proof`, where `X`, `Y`, `Z` are the digits of $(n_0, n_1, n_2)$, for example `rmms_n333.pb.txt` and `rmms_n333_bt_proof` for $\langle 3,3,3 \rangle$.

The main binary here, `rank_lower_bound_verifier_main`, takes `proof_cert/rmms_nXYZ.pb.txt`, and either successfully verifies the lower bound or terminates with an error.

## Verifying $\mathbf{R}(\langle 3,3,3\rangle) \ge 20$ over $\mathbb{F}_2$

All commands below should be run from the **repository root** (the directory containing `MODULE.bazel`).

To verify the proof for the `3×3×3` matrix multiplication tensor:

```bash
bazel build --config=opt //proof_verifier:rank_lower_bound_verifier_main

bazel-bin/proof_verifier/rank_lower_bound_verifier_main proof_cert/rmms_n333.pb.txt
```

It should take about 10 seconds.

On success, the output will end with a line of the form:

```text
Verified. The rank lower bound for 3x3x3 matrix multiplication tensor is 20 over F_2.
```

## Verifying other proofs

Run
```
./proof_verifier/verify.sh [--use-gpu] proof_cert/rmms_nXYZ.pb.txt
```

It change the matrix size in `proof_verifier/dimension.h` before running `rank_lower_bound_verifier_main`. 

`rmms_n344.pb.txt` is stored compressed as `proof_cert/rmms_n344.pb.txt.gz`. Please `gunzip proof_cert/rmms_n344.pb.txt.gz` before the verification.

## (Optional) Inspecting a proof by hand

Proof certificates only store the compact (binary) `compact_restrictions` field. To read the restriction matrices and the restricted tensor for each entry in a human-readable form, populate the `restrictions_text` and `tensor` fields with the `add_verbose_fields_main` tool from `rank_search/` (see `rank_search/README.md`):

```bash
bazel build --config=opt //rank_search:add_verbose_fields_main

bazel-bin/rank_search/add_verbose_fields_main proof_cert/rmms_n333.pb.txt
```

The compiled-in matrix size (`kN0/kN1/kN2` in `proof_verifier/dimension.h`) must match the proof.
