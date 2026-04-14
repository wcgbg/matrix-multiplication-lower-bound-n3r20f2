load("@protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")
load("@protobuf//bazel:py_proto_library.bzl", "py_proto_library")
load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")
load("@rules_cuda//cuda:defs.bzl", "cuda_library")
load("@rules_proto//proto:defs.bzl", "proto_library")

package(default_visibility = ["//visibility:public"])

# When --define=use_gpu=1 is set, use CUDA backend; otherwise use stub (CPU-only).
config_setting(
    name = "use_cuda",
    define_values = {"use_gpu": "1"},
)

proto_library(
    name = "restricted_mm_proto",
    srcs = ["restricted_mm.proto"],
)

cc_proto_library(
    name = "restricted_mm_cc_proto",
    deps = [":restricted_mm_proto"],
)

py_proto_library(
    name = "restricted_mm_py_proto",
    deps = [":restricted_mm_proto"],
)

cc_library(
    name = "static_matrix",
    hdrs = ["static_matrix.h"],
    deps = ["@ng-log//:ng-log"],
)

cc_test(
    name = "static_matrix_test",
    srcs = ["static_matrix_test.cc"],
    deps = [
        ":static_matrix",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "dynamic_matrix",
    srcs = ["dynamic_matrix.cc"],
    hdrs = ["dynamic_matrix.h"],
    deps = ["@ng-log//:ng-log"],
)

cc_test(
    name = "dynamic_matrix_test",
    srcs = ["dynamic_matrix_test.cc"],
    deps = [
        ":dynamic_matrix",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "math_utils",
    hdrs = ["math_utils.h"],
    deps = ["@ng-log//:ng-log"],
)

cc_test(
    name = "math_utils_test",
    srcs = ["math_utils_test.cc"],
    deps = [
        ":math_utils",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "proto_io",
    hdrs = ["proto_io.h"],
    deps = [
        "@ng-log//:ng-log",
        "@protobuf",
    ],
)

cc_library(
    name = "dimension",
    hdrs = ["dimension.h"],
)

cc_library(
    name = "tensor",
    hdrs = ["tensor.h"],
)

cc_library(
    name = "tensor_utils",
    hdrs = ["tensor_utils.h"],
    deps = [
        ":tensor",
        "@boost.algorithm",
        "@ng-log//:ng-log",
    ],
)

cc_test(
    name = "tensor_utils_test",
    srcs = ["tensor_utils_test.cc"],
    deps = [
        ":tensor",
        ":tensor_utils",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "restrictions",
    hdrs = ["restrictions.h"],
    deps = [
        ":math_utils",
        ":restricted_mm_cc_proto",
        ":static_matrix",
        ":tensor_utils",
        "@boost.container",
    ],
)

cc_library(
    name = "backtracking_proof",
    srcs = ["backtracking_proof.cc"],
    hdrs = ["backtracking_proof.h"],
    deps = [
        "@boost.iostreams",
        "@ng-log//:ng-log",
    ],
)

cc_test(
    name = "backtracking_proof_test",
    srcs = ["backtracking_proof_test.cc"],
    deps = [
        ":backtracking_proof",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)

cuda_library(
    name = "rank_lower_bound_gpu_cuda",
    tags = ["cuda"],
    srcs = ["rank_lower_bound_gpu_cuda.cu"],
    hdrs = ["rank_lower_bound_gpu.h"],
    # copts = ["-Wno-deprecated-gpu-targets"],
    deps = [
        ":tensor",
    ],
)

cc_library(
    name = "rank_lower_bound_gpu",
    srcs = select({
        ":use_cuda": [],
        "//conditions:default": ["rank_lower_bound_gpu_stub.cc"],
    }),
    hdrs = ["rank_lower_bound_gpu.h"],
    deps = select({
        ":use_cuda": [
            ":rank_lower_bound_gpu_cuda",
            ":tensor",
        ],
        "//conditions:default": [
            ":tensor",
            "@ng-log//:ng-log",
        ],
    }),
)

cc_library(
    name = "rank_lower_bound_basic_technics",
    hdrs = ["rank_lower_bound_basic_technics.h"],
    deps = [
        ":dynamic_matrix",
        ":rank_lower_bound_gpu",
        ":tensor_utils",
        "@ng-log//:ng-log",
        "@onetbb//:tbb",
    ],
)

cc_test(
    name = "rank_lower_bound_basic_technics_test",
    srcs = ["rank_lower_bound_basic_technics_test.cc"],
    deps = [
        ":dynamic_matrix",
        ":rank_lower_bound_basic_technics",
        ":tensor_utils",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)

cc_binary(
    name = "rank_lower_bound_foreced_product_benchmark_432_main",
    srcs = ["rank_lower_bound_foreced_product_benchmark_432_main.cc"],
    malloc = "@mimalloc",
    deps = [
        ":rank_lower_bound_basic_technics",
        ":tensor_utils",
        "@ng-log//:ng-log",
    ],
)

cc_binary(
    name = "rank_lower_bound_foreced_product_benchmark_443_main",
    srcs = ["rank_lower_bound_foreced_product_benchmark_443_main.cc"],
    malloc = "@mimalloc",
    deps = [
        ":rank_lower_bound_basic_technics",
        ":tensor_utils",
        "@ng-log//:ng-log",
    ],
)

cc_library(
    name = "rank_lower_bound_backtracking_verifier",
    hdrs = ["rank_lower_bound_backtracking_verifier.h"],
    deps = [
        ":backtracking_proof",
        ":restrictions",
        ":static_matrix",
        "@boost.unordered",
        "@ng-log//:ng-log",
    ],
)

cc_library(
    name = "rank_lower_bound_verifier",
    hdrs = ["rank_lower_bound_verifier.h"],
    deps = [
        ":dimension",
        ":rank_lower_bound_backtracking_verifier",
        ":rank_lower_bound_basic_technics",
        ":restricted_mm_cc_proto",
        ":restrictions",
        "@boost.unordered",
        "@ng-log//:ng-log",
        "@onetbb//:tbb",
    ],
)

cc_binary(
    name = "rank_lower_bound_verifier_main",
    srcs = ["rank_lower_bound_verifier_main.cc"],
    malloc = "@mimalloc",
    deps = [
        ":proto_io",
        ":rank_lower_bound_verifier",
        ":restricted_mm_cc_proto",
        "@ng-log//:ng-log",
    ],
)

cc_test(
    name = "restrictions_test",
    srcs = ["restrictions_test.cc"],
    deps = [
        ":restricted_mm_cc_proto",
        ":restrictions",
        ":tensor_utils",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)
