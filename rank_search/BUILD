load(
    "@hedron_compile_commands//:refresh_compile_commands.bzl",
    "refresh_compile_commands",
)
load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "restrictions_set",
    hdrs = ["restrictions_set.h"],
    deps = [
        "//proof_verifier:restrictions",
        "@boost.unordered",
        "@ng-log//:ng-log",
        "@onetbb//:tbb",
    ],
)

cc_library(
    name = "restrictions_enumerator_slow",
    hdrs = ["restrictions_enumerator_slow.h"],
    deps = [
        "//proof_verifier:math_utils",
        "//proof_verifier:restricted_mm_cc_proto",
        "//proof_verifier:restrictions",
        "//proof_verifier:static_matrix",
        "@boost.unordered",
        "@ng-log//:ng-log",
        "@onetbb//:tbb",
    ],
)

cc_test(
    name = "restrictions_enumerator_slow_test",
    srcs = ["restrictions_enumerator_slow_test.cc"],
    data = glob(["testdata/*"]),
    deps = [
        ":restrictions_enumerator_slow",
        "//proof_verifier:proto_io",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
        "@ng-log//:ng-log",
    ],
)

cc_library(
    name = "restrictions_enumerator",
    hdrs = ["restrictions_enumerator.h"],
    deps = [
        ":restrictions_set",
        "//proof_verifier:math_utils",
        "//proof_verifier:restricted_mm_cc_proto",
        "//proof_verifier:restrictions",
        "//proof_verifier:static_matrix",
        "@ng-log//:ng-log",
        "@onetbb//:tbb",
    ],
)

cc_test(
    name = "restrictions_enumerator_test",
    srcs = ["restrictions_enumerator_test.cc"],
    data = glob(["testdata/*"]),
    deps = [
        ":restrictions_enumerator",
        "//proof_verifier:proto_io",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
        "@ng-log//:ng-log",
    ],
)

cc_binary(
    name = "restrictions_enumerator_main",
    srcs = ["restrictions_enumerator_main.cc"],
    malloc = "@mimalloc",
    deps = [
        ":restrictions_enumerator",
        ":restrictions_enumerator_slow",
        "//proof_verifier:dimension",
        "//proof_verifier:proto_io",
        "//proof_verifier:restricted_mm_cc_proto",
        "@ng-log//:ng-log",
    ],
)

cc_library(
    name = "restrictions_map",
    hdrs = ["restrictions_map.h"],
    deps = [
        "//proof_verifier:math_utils",
        "//proof_verifier:restrictions",
        "//proof_verifier:static_matrix",
        "@boost.container",
        "@boost.unordered",
        "@ng-log//:ng-log",
        "@onetbb//:tbb",
    ],
)

cc_test(
    name = "restrictions_map_test",
    srcs = ["restrictions_map_test.cc"],
    deps = [
        ":restrictions_map",
        "//proof_verifier:math_utils",
        "//proof_verifier:restricted_mm_cc_proto",
        "//proof_verifier:restrictions",
        "//proof_verifier:static_matrix",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "rank_lower_bound_backtracking",
    hdrs = ["rank_lower_bound_backtracking.h"],
    deps = [
        ":restrictions_map",
        "//proof_verifier:backtracking_proof",
        "//proof_verifier:restricted_mm_cc_proto",
        "//proof_verifier:restrictions",
        "//proof_verifier:static_matrix",
        "@ng-log//:ng-log",
        "@onetbb//:tbb",
    ],
)

cc_library(
    name = "rank_lower_bound_computer",
    hdrs = ["rank_lower_bound_computer.h"],
    deps = [
        ":rank_lower_bound_backtracking",
        ":restrictions_map",
        "//proof_verifier:backtracking_proof",
        "//proof_verifier:dynamic_matrix",
        "//proof_verifier:math_utils",
        "//proof_verifier:proto_io",
        "//proof_verifier:rank_lower_bound_basic_technics",
        "//proof_verifier:restricted_mm_cc_proto",
        "//proof_verifier:restrictions",
        "//proof_verifier:static_matrix",
        "//proof_verifier:tensor",
        "@ng-log//:ng-log",
        "@onetbb//:tbb",
    ],
)

cc_binary(
    name = "rank_lower_bound_computer_main",
    srcs = ["rank_lower_bound_computer_main.cc"],
    malloc = "@mimalloc",
    deps = [
        ":rank_lower_bound_computer",
        "//proof_verifier:backtracking_proof",
        "//proof_verifier:dimension",
        "//proof_verifier:proto_io",
        "//proof_verifier:restricted_mm_cc_proto",
        "@boost.algorithm",
        "@ng-log//:ng-log",
        "@onetbb//:tbb",
    ],
)

cc_test(
    name = "rank_lower_bound_computer_test",
    srcs = ["rank_lower_bound_computer_test.cc"],
    data = glob(["testdata/*"]),
    deps = [
        ":rank_lower_bound_computer",
        "//proof_verifier:proto_io",
        "//proof_verifier:restricted_mm_cc_proto",
        "@boost.algorithm",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
        "@ng-log//:ng-log",
    ],
)

refresh_compile_commands(
    name = "refresh_compile_commands",
    exclude_external_sources = True,
    targets = [],
)
