// swift-tools-version: 6.2
import PackageDescription
import Foundation

// Resolve TensorFlow C include/lib locations with sensible per-OS defaults,
// while allowing LIBTENSORFLOW_INCLUDEDIR / LIBTENSORFLOW_LIBDIR overrides.
let env = ProcessInfo.processInfo.environment

#if os(Linux)
let fm = FileManager.default
let linuxIncludeCandidates: [String] = [
    env["LIBTENSORFLOW_INCLUDEDIR"],
    "/usr/local/include",
    "/usr/include"
].compactMap { $0 }

let linuxLibCandidates: [String] = [
    env["LIBTENSORFLOW_LIBDIR"],
    "/usr/local/lib",
    "/usr/lib/x86_64-linux-gnu",
    "/usr/lib/aarch64-linux-gnu"
].compactMap { $0 }

let tfInclude: String =
    linuxIncludeCandidates.first { fm.fileExists(atPath: "\($0)/tensorflow/c/c_api.h") }
    ?? (env["LIBTENSORFLOW_INCLUDEDIR"] ?? "/usr/local/include")

let tfLib: String =
    linuxLibCandidates.first { fm.fileExists(atPath: "\($0)/libtensorflow.so") }
    ?? (env["LIBTENSORFLOW_LIBDIR"] ?? "/usr/local/lib")

#else
// macOS defaults (Apple Silicon Homebrew), but allow overrides via ENV
let tfInclude = env["LIBTENSORFLOW_INCLUDEDIR"] ?? "/opt/homebrew/opt/libtensorflow/include"
let tfLib     = env["LIBTENSORFLOW_LIBDIR"]     ?? "/opt/homebrew/opt/libtensorflow/lib"
#endif

let package = Package(
    name: "TF4Swift",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .library(name: "TF4SwiftCore", targets: ["TF4SwiftCore"]),
        .library(name: "TF4SwiftOps",  targets: ["TF4SwiftOps"]),
        .executable(name: "tf4swift-opgen", targets: ["TF4SwiftOpGen"]),
    ],
    .package(
      url: "https://github.com/apple/swift-protobuf.git",
      from: "1.25.0"
    ),
    targets: [
        // C shim over TensorFlow C API
        .target(
            name: "CTensorFlow",
            path: "Sources/CTensorFlow",
            publicHeadersPath: "include",
            cSettings: [
                .headerSearchPath("include"),
                // Linux: rely on discovered/include ENV only
                .unsafeFlags(["-I", tfInclude], .when(platforms: [.linux])),
                // macOS: keep common Homebrew fallbacks in addition to ENV/default
                .unsafeFlags([
                    "-I", tfInclude,
                    "-I", "/opt/homebrew/opt/libtensorflow/include",
                    "-I", "/usr/local/opt/libtensorflow/include"
                ], .when(platforms: [.macOS]))
            ]
        ),

        // Public Swift API
        .target(
            name: "TF4SwiftCore",
            dependencies: ["CTensorFlow"],
            path: "Sources/TF4SwiftCore",
            // Help the Clang importer see TF headers
            swiftSettings: [
                .unsafeFlags(["-Xcc","-I","-Xcc", tfInclude], .when(platforms: [.linux])),
                .unsafeFlags([
                    "-Xcc","-I","-Xcc", tfInclude,
                    "-Xcc","-I","-Xcc","/opt/homebrew/opt/libtensorflow/include",
                    "-Xcc","-I","-Xcc","/usr/local/opt/libtensorflow/include"
                ], .when(platforms: [.macOS]))
            ],
            linkerSettings: [
                .linkedLibrary("tensorflow"),
                // Library search paths
                .unsafeFlags(["-L", tfLib], .when(platforms: [.linux])),
                .unsafeFlags([
                    "-L", tfLib,
                    "-L", "/opt/homebrew/opt/libtensorflow/lib",
                    "-L", "/usr/local/opt/libtensorflow/lib"
                ], .when(platforms: [.macOS]))
            ]
        ),

        // Generated op wrappers (placeholder; keep at least one .swift file in dir)
        .target(
            name: "TF4SwiftOps",
            dependencies: ["TF4SwiftCore"],
            path: "Sources/TF4SwiftOps"
        ),

        // Op generator binary — embed rpaths here so it can find libtensorflow at runtime
        .executableTarget(
            name: "TF4SwiftOpGen",
            dependencies: [
                "TF4SwiftCore",
                // Minimal protobuf decoding of tensorflow.OpList:
                .product(name: "SwiftProtobuf", package: "swift-protobuf"),
                ],
            path: "Sources/TF4SwiftOpGen",
            resources: [
                .copy("ops.pbtxt")
            ],
            linkerSettings: [
                // Linker search paths
                .unsafeFlags(["-L", tfLib], .when(platforms: [.linux])),
                .unsafeFlags([
                    "-L", tfLib,
                    "-L", "/opt/homebrew/opt/libtensorflow/lib",
                    "-L", "/usr/local/opt/libtensorflow/lib"
                ], .when(platforms: [.macOS])),
                // Runtime search paths (rpath)
                .unsafeFlags(["-Xlinker","-rpath","-Xlinker", tfLib], .when(platforms: [.linux])),
                .unsafeFlags([
                    "-Xlinker","-rpath","-Xlinker", tfLib,
                    "-Xlinker","-rpath","-Xlinker","/opt/homebrew/opt/libtensorflow/lib",
                    "-Xlinker","-rpath","-Xlinker","/usr/local/opt/libtensorflow/lib"
                ], .when(platforms: [.macOS]))
            ]
        ),

        // Tests (test runner is a binary => needs rpath too)
        .testTarget(
            name: "TF4SwiftCoreTests",
            dependencies: ["TF4SwiftCore", "TF4SwiftOps"],
            path: "Tests/TF4SwiftCoreTests",
            linkerSettings: [
                .unsafeFlags(["-L", tfLib], .when(platforms: [.linux])),
                .unsafeFlags([
                    "-L", tfLib,
                    "-L", "/opt/homebrew/opt/libtensorflow/lib",
                    "-L", "/usr/local/opt/libtensorflow/lib"
                ], .when(platforms: [.macOS])),
                .unsafeFlags(["-Xlinker","-rpath","-Xlinker", tfLib], .when(platforms: [.linux])),
                .unsafeFlags([
                    "-Xlinker","-rpath","-Xlinker", tfLib,
                    "-Xlinker","-rpath","-Xlinker","/opt/homebrew/opt/libtensorflow/lib",
                    "-Xlinker","-rpath","-Xlinker","/usr/local/opt/libtensorflow/lib"
                ], .when(platforms: [.macOS]))
            ]
        )
    ]
)
