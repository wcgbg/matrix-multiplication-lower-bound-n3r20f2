## Rank Search

This directory contains the search code that discovers computer-verifiable tensor rank lower bounds.

All commands below should be run from the **repository root** (the directory containing `MODULE.bazel`).

### Set matrix size

Modify `kN0`, `kN1`, and `kN2` in `proof_verifier/dimension.h` to set the matrix size if the matrix multiplication problem is not $3 \times 3 \times 3$.

### Enumerate restricted matrix $A$ 

Build and run the enumerator:

```bash
bazel build --config=opt //rank_search:restrictions_enumerator_main

bazel-bin/rank_search/restrictions_enumerator_main
```

On a MacBook Air M4 16 GB this takes about 0.1 seconds on the 3x3x3 case.

It produces `rmms_nXYZ.pb.txt` in the current working directory (the repository root).

Pass `--verbose` to `restrictions_enumerator_main` to write verbose fields to the output protobuf. The file is larger but easier to read.

#### (Optional) Add verbose fields to an existing proof file

If you already have an `rmms_nXYZ.pb.txt` (or `*_updated.pb.txt`) that was produced without `--verbose` (like `proof_cert/*.pb.txt`), you can fill in the human-readable `restrictions_text` and `tensor` fields after the fact:

```bash
bazel build --config=opt //rank_search:add_verbose_fields_main

bazel-bin/rank_search/add_verbose_fields_main rmms_n333.pb.txt
```

This rewrites the `.pb.txt` file in place.

### Run the rank lower bound search

Given a file like `rmms_nXYZ.pb.txt`, build and run the backtracking + substitution search:

```bash
bazel build --config=opt //rank_search:rank_lower_bound_computer_main

bazel-bin/rank_search/rank_lower_bound_computer_main rmms_n333.pb.txt --backtracking_step_limit=10000000
```

On a MacBook Air M4 16 GB this takes about 40 minutes. The command above writes:

- `rmms_n333_updated.pb.txt`: the updated proof instance with the discovered rank lower bounds
- `rmms_n333_updated_bt_proof/`: a directory containing compressed backtracking proof fragments (`*.btp`)

You can change the output location with `--output_path=/path/to/updated.pb.txt`; the backtracking proof directory will then be named `<output_prefix>_bt_proof/`.

The search has several other useful flags:

- `--backtracking_step_limit`: limit on the total number of backtracking steps (across all threads) per one restriction subspace
- `--backtracking_max_map_size`: Max size of the per-thread local cache map used by the backtracking search
- `--rank_lower_bound_min`, `--rank_lower_bound_max`: restrict the range of ranks explored
- `--dim_min`, `--dim_max`: restrict which restriction subspace dimentions are processed
- `--ignore_rank_lower_bound`: clear any existing lower bounds in the input before searching


### Verify the discovered proof

To verify the proof produced in step 2, use the standalone verifier in `proof_verifier` (see `proof_verifier/README.md` for details). For the `3×3×3` case:

```bash
bazel build --config=opt //proof_verifier:rank_lower_bound_verifier_main

bazel-bin/proof_verifier/rank_lower_bound_verifier_main rmms_n333_updated.pb.txt
```

On a MacBook Air M4 16 GB this takes about 3 seconds.
