# Proof Certificates

This directory holds machine-checkable rank lower-bound certificates for small matrix multiplication tensors over $\mathbb{F}_2$. Each case is a text protobuf `proof_cert/rmms_nXYZ.pb.txt` together with a directory `proof_cert/rmms_nXYZ_bt_proof/` of compressed backtracking fragments (`.btp`). The digits `X`, `Y`, `Z` in the filename are $(n_0, n_1, n_2)$ for the format $\langle n_0, n_1, n_2 \rangle$.

The sections below record the commands used to **discover** each bound. The checked-in files here are the resulting certificates. To **verify** a certificate, see [proof_verifier/README.md](../proof_verifier/README.md). For the general search workflow and flag descriptions, see [rank_search/README.md](../rank_search/README.md).

The commands below should be run from the **repository root**—the directory that contains `MODULE.bazel`.

## Reproducing a Certificate

## $\mathbf{R}(\langle 2,2,2\rangle) \ge 7$

Set `kN0=2`, `kN1=2`, and `kN2=2` in `proof_verifier/dimension.h`.
```
bazel build --config=opt //rank_search:restrictions_enumerator_main
bazel-bin/rank_search/restrictions_enumerator_main
bazel build --config=opt //rank_search:rank_lower_bound_computer_main
bazel-bin/rank_search/rank_lower_bound_computer_main rmms_n222.pb.txt
```

## $\mathbf{R}(\langle 2,2,3\rangle) \ge 11$
Set `kN0=2`, `kN1=2`, and `kN2=3` in `proof_verifier/dimension.h`.
```
bazel build --config=opt //rank_search:restrictions_enumerator_main
bazel-bin/rank_search/restrictions_enumerator_main
bazel build --config=opt //rank_search:rank_lower_bound_computer_main
bazel-bin/rank_search/rank_lower_bound_computer_main rmms_n223.pb.txt
```

## $\mathbf{R}(\langle 2,2,4\rangle) \ge 14$
Set `kN0=2`, `kN1=2`, and `kN2=4` in `proof_verifier/dimension.h`.
```
bazel build --config=opt //rank_search:restrictions_enumerator_main
bazel-bin/rank_search/restrictions_enumerator_main
bazel build --config=opt //rank_search:rank_lower_bound_computer_main
bazel-bin/rank_search/rank_lower_bound_computer_main rmms_n224.pb.txt
```

## $\mathbf{R}(\langle 2,3,3\rangle) \ge 15$
Set `kN0=2`, `kN1=3`, and `kN2=3` in `proof_verifier/dimension.h`.
```
bazel build --config=opt //rank_search:restrictions_enumerator_main
bazel-bin/rank_search/restrictions_enumerator_main
bazel build --config=opt //rank_search:rank_lower_bound_computer_main
bazel-bin/rank_search/rank_lower_bound_computer_main rmms_n233.pb.txt --backtracking_step_limit=100000
```

## $\mathbf{R}(\langle 3,2,4\rangle) \ge 19$
Set `kN0=3`, `kN1=2`, and `kN2=4` in `proof_verifier/dimension.h`.
```
bazel build --config=opt //rank_search:restrictions_enumerator_main
bazel-bin/rank_search/restrictions_enumerator_main
bazel build --config=opt //rank_search:rank_lower_bound_computer_main
bazel-bin/rank_search/rank_lower_bound_computer_main rmms_n324.pb.txt --backtracking_step_limit=10000000
```

`rank_lower_bound_computer_main` took 110 minutes on MacBook Air M4 (8 cores, 16 GB RAM).

## $\mathbf{R}(\langle 3,3,3\rangle) \ge 20$

Set `kN0=3`, `kN1=3`, and `kN2=3` in `proof_verifier/dimension.h`.
```
bazel build --config=opt //rank_search:restrictions_enumerator_main
bazel-bin/rank_search/restrictions_enumerator_main
bazel build --config=opt //rank_search:rank_lower_bound_computer_main
bazel-bin/rank_search/rank_lower_bound_computer_main rmms_n333.pb.txt --backtracking_step_limit=10000000
```

`rank_lower_bound_computer_main` took 40 minutes on MacBook Air M4 (8 cores, 16 GB RAM).

## $\mathbf{R}(\langle 3,3,4\rangle) \ge 25$

The search `rank_lower_bound_computer_main` is split into two passes by restriction subspace dimension (`--dim_min` / `--dim_max`) to reduce peak memory.

Set `kN0=3`, `kN1=3`, and `kN2=4` in `proof_verifier/dimension.h`.
```
bazel build --config=opt //rank_search:restrictions_enumerator_main
bazel-bin/rank_search/restrictions_enumerator_main
bazel build --config=opt //rank_search:rank_lower_bound_computer_main
bazel-bin/rank_search/rank_lower_bound_computer_main rmms_n334.pb.txt --backtracking_step_limit=100000000 --dim_min=4
mv rmms_n334_updated.pb.txt rmms_n334.pb.txt
bazel-bin/rank_search/rank_lower_bound_computer_main rmms_n334.pb.txt --backtracking_step_limit=100000000 --dim_max=3 --backtracking_max_map_size=3000000
```

On AWS `c8g.16xlarge` (64 Graviton cores, 128 GB RAM), the first pass took 4 days and 9 hours and the second pass took 4.7 hours.

## $\mathbf{R}(\langle 3,4,4\rangle) \ge 29$


Set `kN0=3`, `kN1=4`, and `kN2=4` in `proof_verifier/dimension.h`.
```
bazel build --config=opt //rank_search:restrictions_enumerator_main
bazel-bin/rank_search/restrictions_enumerator_main
bazel build --config=opt //rank_search:rank_lower_bound_computer_main
bazel-bin/rank_search/rank_lower_bound_computer_main rmms_n344.pb.txt --backtracking_step_limit=400000
```

On AWS `c8g.48xlarge` (192 Graviton cores, 384 GB RAM), `restrictions_enumerator_main` took 3.3 minutes, and `rank_lower_bound_computer_main` took 4.8 hours.

The checked-in `rmms_n344` certificate is stored compressed as `proof_cert/rmms_n344.pb.txt.gz`.
