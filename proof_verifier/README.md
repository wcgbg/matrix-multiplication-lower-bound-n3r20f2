# Proof Verifier

This directory contains a standalone verifier for the computer-generated tensor rank lower bound proofs.

Proof instances live in the top-level `proof/` directory as text-format protobufs named `proof/rmms_nXYZ.pb.txt` and binary files named `proof/rmms_nXYZ_bt_proof`, where `X`, `Y`, `Z` are the digits of $(n_0, n_1, n_2)$, for example `rmms_n333.pb.txt` and `rmms_n333_bt_proof` for $\langle 3,3,3 \rangle$.

The main binary here, `rank_lower_bound_verifier_main`, takes `proof/rmms_nXYZ.pb.txt`, and either successfully verifies the lower bound or terminates with an error.

## Verifying $\mathbf{R}(\langle 3,3,3\rangle) \ge 20$ over $\mathbb{F}_2$

All commands below should be run from the **repository root** (the directory containing `MODULE.bazel`).

To verify the proof for the `3×3×3` matrix multiplication tensor:

```bash
bazel build -c opt //proof_verifier:rank_lower_bound_verifier_main

bazel-bin/proof_verifier/rank_lower_bound_verifier_main proof/rmms_n333.pb.txt
```

It should take about 10 seconds.

On success, the output will end with a line of the form:

```text
Verified. The rank lower bound for 3x3x3 matrix multiplication tensor is 20 over F_2.
```

## Verifying other proofs

Run
```
./proof_verifier/verify.sh proof/rmms_nXYZ.pb.txt
```

It change the matrix size in `proof_verifier/dimension.h` before running `rank_lower_bound_verifier_main`. 
