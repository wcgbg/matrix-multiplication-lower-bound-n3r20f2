load("@protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")
load("@protobuf//bazel:py_proto_library.bzl", "py_proto_library")
load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")
load("@rules_proto//proto:defs.bzl", "proto_library")

package(default_visibility = ["//visibility:public"])

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
    deps = [
        "@boost.algorithm",
        "@ng-log//:ng-log",
    ],
)

cc_test(
    name = "tensor_test",
    srcs = ["tensor_test.cc"],
    deps = [
        ":tensor",
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
        ":tensor",
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

cc_library(
    name = "rank_lower_bound_basic_technics",
    hdrs = ["rank_lower_bound_basic_technics.h"],
    deps = [
        ":dynamic_matrix",
        ":tensor",
        "@ng-log//:ng-log",
        "@onetbb//:tbb",
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
        ":tensor",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)
