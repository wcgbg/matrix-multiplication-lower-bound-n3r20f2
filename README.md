# Automated Lower Bounds for Small Matrix Multiplication Complexity over Finite Fields

We develop an automated framework for proving lower bounds on the bilinear complexity of matrix multiplication over finite fields. Our approach systematically combines symmetry reduction, restriction space enumeration, and recursive substitution arguments, culminating in efficiently verifiable proof certificates. Using this framework, we obtain several new lower bounds for various small matrix formats. Most notably, we prove that the bilinear complexity of multiplying two $3 \times 3$ matrices over $\mathbb{F}_2$ is at least $20$, improving upon the longstanding lower bound of $19$ (Bläser 2003). Our computer search discovers this proof in approximately $1.5$ hours on a laptop, and the resulting certificate can be verified in seconds.

For details, see out [paper](https://arxiv.org/abs/2603.07280).

## Results

$\mathbf{R}(\langle 3,3,3\rangle) \ge 20$ over $\mathbb{F}_2$. The tensor rank of the matrix multiplication tensor for two $3 \times 3$ matrices is at least 20 over $\mathbb{F}_2$; equivalently, any bilinear algorithm for multiplying two $3 \times 3$ matrices over $\mathbb{F}_2$ must use at least 20 scalar multiplications. This improves upon the longstanding lower bound of $19$ established by [Bläser03](https://www.sciencedirect.com/science/article/pii/S0885064X02000079).

$\mathbf{R}(\langle 2,3,4\rangle) \ge 19$ over $\mathbb{F}_2$. It completes the missing proof in [HK71](https://epubs.siam.org/doi/abs/10.1137/0120004).

$\mathbf{R}(\langle 3,3,4\rangle) \ge 25$ over $\mathbb{F}_2$.

$\mathbf{R}(\langle 3,4,4\rangle) \ge 29$ over $\mathbb{F}_2$.


## Build

This Git repository requires [LFS](https://git-lfs.com/). Install it before `git clone`.

We use Bazel (via Bazelisk) to build the project. First, install Bazelisk from `https://github.com/bazelbuild/bazelisk`.

This project requires C++20 support. If needed, configure your C++ toolchain in the first two lines of `.bazelrc`. If you are using Apple Clang, uncomment lines 7–10 in `.bazelrc`.

To build all without CUDA:
```
bazel build -c opt --build_tag_filters=-cuda //...
```

To build all with CUDA:
```
bazel build -c opt --define=use_gpu=1 //...
```
The CUDA build has been tested on Ubuntu 24.04, GCC 14, and CUDA 12.8.

## Code Layout

To verify the proof, see [proof_verifier/README.md](proof_verifier/README.md).

To understand how we discover the rank lower bound and construct the proof, see [rank_search/README.md](rank_search/README.md).

## Citation
```bibtex
@misc{wang2026complexitylowerboundssmall,
      title={Automated Lower Bounds for Small Matrix Multiplication Complexity over Finite Fields}, 
      author={Chengu Wang},
      year={2026},
      eprint={2603.07280},
      archivePrefix={arXiv},
      primaryClass={cs.CC},
      url={https://arxiv.org/abs/2603.07280}, 
}
```
