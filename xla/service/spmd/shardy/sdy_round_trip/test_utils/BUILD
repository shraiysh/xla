load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([":friends"]),
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
    ],
)

cc_library(
    name = "stablehlo_to_hlo_to_stablehlo",
    srcs = ["stablehlo_to_hlo_to_stablehlo.cc"],
    hdrs = ["stablehlo_to_hlo_to_stablehlo.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/hlo/translate:stablehlo",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_proto_cc",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:UBDialect",
        "@tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "testing_pipeline",
    srcs = ["testing_pipeline.cc"],
    hdrs = ["testing_pipeline.h"],
    deps = [
        ":stablehlo_to_hlo_to_stablehlo",
        "//xla/service/spmd/shardy/sdy_round_trip:pipelines",
        "@llvm-project//mlir:Pass",
    ],
)
