# Complexity Lower Bounds of Small Matrix Multiplication over Finite Fields via Backtracking and Substitution

We introduce a new method for proving bilinear complexity lower bounds for matrix multiplication over finite fields. The approach combines the substitution method with a systematic backtracking search over linear restrictions on the first matrix $A$ in the product $AB = C^T$. We enumerate restriction classes up to symmetry; for each class we either obtain a rank lower bound by classical arguments or branch further via the substitution method. The search is organized by dynamic programming on the restricted matrix $A$. As an application we prove that the bilinear complexity of multiplying two $3 \times 3$ matrices over $\mathbb{F}_2$ is at least $20$, improving the longstanding lower bound of $19$ (Bläser 2003). The proof is found automatically within 1.5 hours on a laptop and verified in seconds.

For details, see out [paper](paper/main.pdf).

## Results

$\mathbf{R}(\langle 3,3,3\rangle) \ge 20$ over $\mathbb{F}_2$, i.e. the tensor rank of the matrix multiplication tensor between two $3 \times 3$ matrices is at least 20 over field $\mathbb{F}_2$. It implies that, for multiplying two $3 \times 3$ matrices over the field $\mathbb{F}_2$, any bilinear algorithm must use at least 20 scalar multiplications in $\mathbb{F}_2$. It improves the previous lower bound of 19 in [Bläser03](https://www.sciencedirect.com/science/article/pii/S0885064X02000079).

$\mathbf{R}(\langle 2,3,4\rangle) \ge 19$ over $\mathbb{F}_2$. It completes the missing proof in [HK71](https://epubs.siam.org/doi/abs/10.1137/0120004).

## Build

We use Bazel (via Bazelisk) to build the project. First, install Bazelisk from `https://github.com/bazelbuild/bazelisk`.

This project requires C++20 support. If needed, configure your C++ toolchain in the first two lines of `.bazelrc`. If you are using Apple Clang, uncomment lines 7–10 in `.bazelrc`.

To build everything:
```
bazel bulid -c opt //...
```

## Code Layout

To verify the proof, see [proof_verifier/README.md](proof_verifier/README.md).

To understand how we discover the rank lower bound and construct the proof, see [rank_search/README.md](rank_search/README.md).
