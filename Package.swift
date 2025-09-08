// swift-tools-version: 6.2
import PackageDescription
import Foundation

#if os(Linux)
import Glibc
#else
import Darwin
#endif

// ------------------------------
// Helpers (no FileManager usage)
// ------------------------------
@inline(__always)
func pathExists(_ p: String) -> Bool {
  p.withCString { access($0, F_OK) == 0 }
}

#if os(Linux)
@inline(__always)
func hasTFHeader(_ dir: String) -> Bool { pathExists("\(dir)/tensorflow/c/c_api.h") }

@inline(__always)
func hasTFLib(_ dir: String) -> Bool { pathExists("\(dir)/libtensorflow.so") }
#else
@inline(__always)
func hasTFHeader(_ dir: String) -> Bool { pathExists("\(dir)/tensorflow/c/c_api.h") }

@inline(__always)
func hasTFLib(_ dir: String) -> Bool {
  // Accept any of these names; Homebrew currently provides the full versioned one.
  pathExists("\(dir)/libtensorflow.dylib")
  || pathExists("\(dir)/libtensorflow.2.dylib")
  || pathExists("\(dir)/libtensorflow.2.19.0.dylib")
}
#endif

let env = ProcessInfo.processInfo.environment

// --------------------------------------
// Compute include/lib paths (deduplicated)
// --------------------------------------
#if os(Linux)
let includeCandidates: [String] = [
  env["LIBTENSORFLOW_INCLUDEDIR"],
  "/usr/local/include",
  "/usr/include"
].compactMap { $0 }

let libCandidates: [String] = [
  env["LIBTENSORFLOW_LIBDIR"],
  "/usr/local/lib",
  "/usr/lib/x86_64-linux-gnu",
  "/usr/lib/aarch64-linux-gnu"
].compactMap { $0 }
#else
// macOS (Apple Silicon Homebrew defaults + /usr/local fallback), overridable via env
let includeCandidates: [String] = [
  env["LIBTENSORFLOW_INCLUDEDIR"],
  "/opt/homebrew/opt/libtensorflow/include",
  "/usr/local/opt/libtensorflow/include"
].compactMap { $0 }

let libCandidates: [String] = [
  env["LIBTENSORFLOW_LIBDIR"],
  "/opt/homebrew/opt/libtensorflow/lib",
  "/usr/local/opt/libtensorflow/lib"
].compactMap { $0 }
#endif

var includePaths: [String] = []
for p in includeCandidates where hasTFHeader(p) && !includePaths.contains(p) { includePaths.append(p) }

var libPaths: [String] = []
for p in libCandidates where hasTFLib(p) && !libPaths.contains(p) { libPaths.append(p) }

// Fallbacks if nothing was detected (keeps old behavior but may emit warnings)
#if os(Linux)
if includePaths.isEmpty { includePaths.append(env["LIBTENSORFLOW_INCLUDEDIR"] ?? "/usr/local/include") }
if libPaths.isEmpty     { libPaths.append(env["LIBTENSORFLOW_LIBDIR"]     ?? "/usr/local/lib") }
#else
if includePaths.isEmpty { includePaths.append(env["LIBTENSORFLOW_INCLUDEDIR"] ?? "/opt/homebrew/opt/libtensorflow/include") }
if libPaths.isEmpty     { libPaths.append(env["LIBTENSORFLOW_LIBDIR"]     ?? "/opt/homebrew/opt/libtensorflow/lib") }
#endif

// ----------------------
// Build/linker settings
// ----------------------
let cIncludeFlags: [String] = includePaths.flatMap { ["-I", $0] }
let swiftImporterIncludeFlags: [SwiftSetting] =
  includePaths.flatMap { [SwiftSetting.unsafeFlags(["-Xcc","-I","-Xcc",$0])] }

let linkSearchFlags: [LinkerSetting] =
  libPaths.flatMap { [LinkerSetting.unsafeFlags(["-L", $0])] }

let rpathFlags: [LinkerSetting] =
  libPaths.flatMap { [LinkerSetting.unsafeFlags(["-Xlinker","-rpath","-Xlinker",$0])] }

// ----------------------
// Package declaration
// ----------------------
let package = Package(
  name: "TF4Swift",
  platforms: [.macOS(.v13)],
  products: [
    .library(name: "TF4SwiftCore", targets: ["TF4SwiftCore"]),
    .library(name: "TF4SwiftOps",  targets: ["TF4SwiftOps"]),
    .executable(name: "tf4swift-opgen", targets: ["TF4SwiftOpGen"]),
  ],
  // No external dependencies needed now
  dependencies: [],
  targets: [
    // C shim over TensorFlow C API
    .target(
      name: "CTensorFlow",
      path: "Sources/CTensorFlow",
      publicHeadersPath: "include",
      cSettings: [
        .headerSearchPath("include"),
        .unsafeFlags(cIncludeFlags)
      ]
    ),

    // Public Swift API
    .target(
      name: "TF4SwiftCore",
      dependencies: ["CTensorFlow"],
      path: "Sources/TF4SwiftCore",
      swiftSettings: swiftImporterIncludeFlags,
      linkerSettings: [
        .linkedLibrary("tensorflow")
      ] + linkSearchFlags
    ),

    // Generated op wrappers
    .target(
      name: "TF4SwiftOps",
      dependencies: ["TF4SwiftCore"],
      path: "Sources/TF4SwiftOps"
    ),

    // Op generator binary — embed rpaths so libtensorflow is found at runtime
    .executableTarget(
      name: "TF4SwiftOpGen",
      dependencies: ["TF4SwiftCore"],
      path: "Sources/TF4SwiftOpGen",
      resources: [.copy("ops.pbtxt")],
      linkerSettings: linkSearchFlags + rpathFlags
    ),

    // Tests (binary, so inherit rpaths too)
    .testTarget(
      name: "TF4SwiftCoreTests",
      dependencies: ["TF4SwiftCore", "TF4SwiftOps"],
      path: "Tests/TF4SwiftCoreTests",
      linkerSettings: linkSearchFlags + rpathFlags
    )
  ]
)
