// swift-tools-version: 6.2
import PackageDescription
import Foundation

// Allow overrides via ENV; default to Homebrew paths on macOS (Apple Silicon).
let env = ProcessInfo.processInfo.environment
let tfInclude = env["LIBTENSORFLOW_INCLUDEDIR"] ?? "/opt/homebrew/opt/libtensorflow/include"
let tfLib     = env["LIBTENSORFLOW_LIBDIR"]     ?? "/opt/homebrew/opt/libtensorflow/lib"

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
    targets: [
        // C shim over TensorFlow C & Eager APIs
        .target(
            name: "CTensorFlow",
            dependencies: [],
            path: "Sources/CTensorFlow",
            publicHeadersPath: "include",
            cSettings: [
                .headerSearchPath("include"),
                // Ensure Clang can find TensorFlow headers
                .unsafeFlags([
                    "-I", tfInclude,
                    "-I", "/opt/homebrew/opt/libtensorflow/include", // Apple Silicon brew
                    "-I", "/usr/local/opt/libtensorflow/include"     // Intel macOS brew
                ], .when(platforms: [.macOS, .linux]))
            ]
        ),

        // Public Swift API over the C layer
        .target(
            name: "TF4SwiftCore",
            dependencies: ["CTensorFlow"],
            path: "Sources/TF4SwiftCore",
            // Help Clang importer (when compiling Swift) see TF headers too
            swiftSettings: [
                .unsafeFlags([
                    "-Xcc","-I", "-Xcc", tfInclude,
                    "-Xcc","-I", "-Xcc", "/opt/homebrew/opt/libtensorflow/include",
                    "-Xcc","-I", "-Xcc", "/usr/local/opt/libtensorflow/include"
                ], .when(platforms: [.macOS, .linux]))
            ],
            linkerSettings: [
                // Link against libtensorflow; keep rpaths off libraries.
                .linkedLibrary("tensorflow"),
                .unsafeFlags([
                    "-L", tfLib,
                    "-L", "/opt/homebrew/opt/libtensorflow/lib",
                    "-L", "/usr/local/opt/libtensorflow/lib"
                ], .when(platforms: [.macOS, .linux]))
            ]
        ),

        // Placeholder for generated op wrappers (keep at least one .swift file here)
        .target(
            name: "TF4SwiftOps",
            dependencies: ["TF4SwiftCore"],
            path: "Sources/TF4SwiftOps"
        ),

        // Op generator tool (final binary -> embed rpaths here)
        .executableTarget(
            name: "TF4SwiftOpGen",
            dependencies: ["TF4SwiftCore"],
            path: "Sources/TF4SwiftOpGen",
            resources: [
                // Make ops.pbtxt available at runtime as Bundle.module resource
                .copy("ops.pbtxt")
            ],
            linkerSettings: [
                // Search paths for the linker
                .unsafeFlags([
                    "-L", tfLib,
                    "-L", "/opt/homebrew/opt/libtensorflow/lib",
                    "-L", "/usr/local/opt/libtensorflow/lib"
                ], .when(platforms: [.macOS, .linux])),

                // Embed rpaths (swiftc needs -Xlinker form)
                .unsafeFlags([
                    "-Xlinker","-rpath","-Xlinker", tfLib,
                    "-Xlinker","-rpath","-Xlinker","/opt/homebrew/opt/libtensorflow/lib",
                    "-Xlinker","-rpath","-Xlinker","/usr/local/opt/libtensorflow/lib"
                ], .when(platforms: [.macOS]))
            ]
        ),

        // Tests (runner is a binary -> embed rpaths here too)
        .testTarget(
            name: "TF4SwiftCoreTests",
            dependencies: ["TF4SwiftCore", "TF4SwiftOps"],
            path: "Tests/TF4SwiftCoreTests",
            linkerSettings: [
                .unsafeFlags([
                    "-L", tfLib,
                    "-L", "/opt/homebrew/opt/libtensorflow/lib",
                    "-L", "/usr/local/opt/libtensorflow/lib"
                ], .when(platforms: [.macOS, .linux])),

                .unsafeFlags([
                    "-Xlinker","-rpath","-Xlinker", tfLib,
                    "-Xlinker","-rpath","-Xlinker","/opt/homebrew/opt/libtensorflow/lib",
                    "-Xlinker","-rpath","-Xlinker","/usr/local/opt/libtensorflow/lib"
                ], .when(platforms: [.macOS]))
            ]
        ),

        // Add this alongside your existing testTarget("TF4SwiftCoreTests", ...)
.testTarget(
    name: "TF4SwiftOpGenTests",
    dependencies: ["TF4SwiftOpGen"],
    path: "Tests/TF4SwiftOpGenTests"
),



    ]
)
